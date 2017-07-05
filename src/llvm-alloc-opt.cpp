// This file is a part of Julia. License is MIT: https://julialang.org/license

#define DEBUG_TYPE "alloc_opt"
#undef DEBUG
#include "llvm-version.h"

#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Pass.h>
#include <llvm/Support/Debug.h>
#include "fix_llvm_assert.h"

#include "codegen_shared.h"
#include "julia.h"
#include "julia_internal.h"

#include <map>

using namespace llvm;

extern std::pair<MDNode*,MDNode*> tbaa_make_child(const char *name, MDNode *parent=nullptr, bool isConstant=false);

namespace {

static void copyMetadata(Instruction *dest, const Instruction *src)
{
#if JL_LLVM_VERSION < 40000
    if (!src->hasMetadata())
        return;
    SmallVector<std::pair<unsigned,MDNode*>,4> TheMDs;
    src->getAllMetadataOtherThanDebugLoc(TheMDs);
    for (const auto &MD : TheMDs)
        dest->setMetadata(MD.first, MD.second);
    dest->setDebugLoc(src->getDebugLoc());
#else
    dest->copyMetadata(*src);
#endif
}

/**
 * Promote `julia.gc_alloc_obj` which do not have escaping root to a alloca and
 * lower other ones to real GC allocation.
 */

constexpr ssize_t invalid_offset = -15;

struct AllocOpt : public FunctionPass {
    static char ID;
    AllocOpt(bool opt=true)
        : FunctionPass(ID),
          optimize(opt)
    {}

private:
    bool optimize;
    LLVMContext *ctx;

    const DataLayout *DL;

    Function *alloc_obj;
    Function *pool_alloc;
    Function *big_alloc;
    Function *ptr_from_objref;

    Type *T_int8;
    Type *T_int32;
    Type *T_size;
    Type *T_pint8;
    Type *T_pjlvalue;
    Type *T_pjlvalue0;
    Type *T_pjlvalue_der;
    Type *T_ppjlvalue0;
    Type *T_ppjlvalue_der;

    MDNode *tbaa_tag;

    bool doInitialization(Module &m) override;
    bool runOnFunction(Function &F) override;
    bool checkUses(Instruction *I, size_t offset, bool &ignore_tag);
    bool checkInst(Instruction *I, Instruction *parent, size_t offset, bool &ignore_tag);
    void replaceUsesWith(Instruction *orig_i, Instruction *new_i, bool ignore_tag);
    void lowerAlloc(CallInst *I, size_t sz);
};

bool AllocOpt::doInitialization(Module &M)
{
    ctx = &M.getContext();
    DL = &M.getDataLayout();

    alloc_obj = M.getFunction("julia.gc_alloc_obj");
    if (!alloc_obj)
        return false;

    ptr_from_objref = M.getFunction("julia.pointer_from_objref");

    T_pjlvalue = alloc_obj->getReturnType();
    T_pjlvalue0 = PointerType::get(cast<PointerType>(T_pjlvalue)->getElementType(), 0);
    T_pjlvalue_der = PointerType::get(cast<PointerType>(T_pjlvalue)->getElementType(),
                                      AddressSpace::Derived);
    T_ppjlvalue0 = PointerType::get(T_pjlvalue, 0);
    T_ppjlvalue_der = PointerType::get(T_pjlvalue, AddressSpace::Derived);

    T_int8 = Type::getInt8Ty(*ctx);
    T_int32 = Type::getInt32Ty(*ctx);
    T_size = sizeof(void*) == 8 ? Type::getInt64Ty(*ctx) : T_int32;
    T_pint8 = PointerType::get(T_int8, 0);

    if (!(pool_alloc = M.getFunction("jl_gc_pool_alloc"))) {
        std::vector<Type*> alloc_pool_args(0);
        alloc_pool_args.push_back(T_pint8);
        alloc_pool_args.push_back(T_int32);
        alloc_pool_args.push_back(T_int32);
        pool_alloc = Function::Create(FunctionType::get(T_pjlvalue, alloc_pool_args, false),
                                      Function::ExternalLinkage, "jl_gc_pool_alloc", &M);
    }
    if (!(big_alloc = M.getFunction("jl_gc_big_alloc"))) {
        std::vector<Type*> alloc_big_args(0);
        alloc_big_args.push_back(T_pint8);
        alloc_big_args.push_back(T_size);
        big_alloc = Function::Create(FunctionType::get(T_pjlvalue, alloc_big_args, false),
                                     Function::ExternalLinkage, "jl_gc_big_alloc", &M);
    }
    MDNode *tbaa_data;
    MDNode *tbaa_data_scalar;
    std::tie(tbaa_data, tbaa_data_scalar) = tbaa_make_child("jtbaa_data");
    tbaa_tag = tbaa_make_child("jtbaa_tag", tbaa_data_scalar).first;

    return true;
}

bool AllocOpt::checkUses(Instruction *I, size_t offset, bool &ignore_tag)
{
    for (auto user: I->users()) {
        auto inst = dyn_cast<Instruction>(user);
        if (!inst || !checkInst(inst, I, offset, ignore_tag)) {
            return false;
        }
    }
    return true;
}

bool AllocOpt::checkInst(Instruction *I, Instruction *parent, size_t offset, bool &ignore_tag)
{
    if (isa<LoadInst>(I))
        return true;
    if (auto call = dyn_cast<CallInst>(I)) {
        if (ptr_from_objref && ptr_from_objref == call->getCalledFunction())
            return true;
        // Only use in argument counts, uses in operand bundle doesn't since it cannot escape.
        for (auto &arg: call->arg_operands()) {
            if (dyn_cast<Instruction>(&arg) == parent) {
                return false;
            }
        }
        if (call->getNumOperandBundles() != 1)
            return false;
        auto obuse = call->getOperandBundleAt(0);
        if (obuse.getTagName() != "jl_roots")
            return false;
        return true;
    }
    if (isa<AddrSpaceCastInst>(I) || isa<BitCastInst>(I))
        return checkUses(I, offset, ignore_tag);
    if (auto gep = dyn_cast<GetElementPtrInst>(I)) {
        APInt apoffset(sizeof(void*) * 8, offset, true);
        if (ignore_tag && (!gep->accumulateConstantOffset(*DL, apoffset) ||
                           apoffset.isNegative()))
            ignore_tag = false;
        return checkUses(I, offset, ignore_tag);
    }
    if (auto store = dyn_cast<StoreInst>(I)) {
        auto storev = store->getValueOperand();
        // Only store value count
        if (storev == parent)
            return false;
        // There's GC root in this object.
        if (auto ptrtype = dyn_cast<PointerType>(storev->getType())) {
            if (ptrtype->getAddressSpace() == AddressSpace::Tracked) {
                return false;
            }
        }
        return true;
    }
    return false;
}

// Both arguments should be pointer of the same type but possibly different address spaces
// `orig_i` is always in addrspace 0.
// This function needs to handle all cases `AllocOpt::checkInst` can handle.
void AllocOpt::replaceUsesWith(Instruction *orig_i, Instruction *new_i, bool ignore_tag)
{
    Type *orig_t = orig_i->getType();
    Type *new_t = new_i->getType();
    if (orig_t == new_t) {
        orig_i->replaceAllUsesWith(new_i);
        orig_i->eraseFromParent();
        return;
    }
    SmallVector<User*, 4> users(orig_i->user_begin(), orig_i->user_end());
    for (auto user: users) {
        if (isa<LoadInst>(user) || isa<StoreInst>(user)) {
            user->replaceUsesOfWith(orig_i, new_i);
        }
        else if (auto call = dyn_cast<CallInst>(user)) {
            if (ptr_from_objref && ptr_from_objref == call->getCalledFunction()) {
                call->replaceAllUsesWith(new_i);
                call->eraseFromParent();
                continue;
            }
            // remove from operand bundle
            user->replaceUsesOfWith(orig_i, ConstantPointerNull::get(cast<PointerType>(new_t)));
        }
        else if (isa<AddrSpaceCastInst>(user) || isa<BitCastInst>(user)) {
            auto I = cast<Instruction>(user);
            auto cast_t = PointerType::get(cast<PointerType>(I->getType())->getElementType(), 0);
            auto replace_i = new_i;
            if (cast_t != orig_t)
                replace_i = new BitCastInst(replace_i, cast_t, "", I);
            replaceUsesWith(I, replace_i, ptr_from_objref);
        }
        else if (auto gep = dyn_cast<GetElementPtrInst>(user)) {
            Instruction *new_gep;
            SmallVector<Value *, 4> IdxOperands(gep->idx_begin(), gep->idx_end());
            if (gep->isInBounds()) {
                new_gep = GetElementPtrInst::CreateInBounds(gep->getSourceElementType(),
                                                            new_i, IdxOperands,
                                                            gep->getName(), gep);
            }
            else {
                new_gep = GetElementPtrInst::Create(gep->getSourceElementType(),
                                                    new_i, IdxOperands,
                                                    gep->getName(), gep);
            }
            copyMetadata(new_gep, gep);
            replaceUsesWith(gep, new_gep, ptr_from_objref);
        }
        else {
            abort();
        }
    }
    assert(orig_i->user_empty());
    orig_i->eraseFromParent();
}

void AllocOpt::lowerAlloc(CallInst *I, size_t sz)
{
    int osize;
    int offset = jl_gc_classify_pools(sz, &osize);
    auto ptls = I->getArgOperand(0);
    CallInst *newI;
    if (offset < 0) {
        newI = CallInst::Create(big_alloc, {ptls, ConstantInt::get(T_size, sz + sizeof(void*))},
                                None, "", I);
    }
    else {
        auto pool_offs = ConstantInt::get(T_int32, offset);
        auto pool_osize = ConstantInt::get(T_int32, osize);
        newI = CallInst::Create(pool_alloc, {ptls, pool_offs, pool_osize}, None, "", I);
    }
    auto tag = I->getArgOperand(2);
    copyMetadata(newI, I);
    const auto &dbg = I->getDebugLoc();
    auto derived = new AddrSpaceCastInst(newI, T_pjlvalue_der, "", I);
    derived->setDebugLoc(dbg);
    auto cast = new BitCastInst(derived, T_ppjlvalue_der, "", I);
    cast->setDebugLoc(dbg);
    auto tagaddr = GetElementPtrInst::Create(T_pjlvalue, cast, {ConstantInt::get(T_size, -1)},
                                             "", I);
    tagaddr->setDebugLoc(dbg);
    auto store = new StoreInst(tag, tagaddr, I);
    store->setMetadata(LLVMContext::MD_tbaa, tbaa_tag);
    store->setDebugLoc(dbg);
    I->replaceAllUsesWith(newI);
    I->eraseFromParent();
}

bool AllocOpt::runOnFunction(Function &F)
{
    if (!alloc_obj)
        return false;
    std::map<CallInst*,size_t> allocs;
    for (auto &bb: F) {
        for (auto &I: bb) {
            auto call = dyn_cast<CallInst>(&I);
            if (!call)
                continue;
            auto callee = call->getCalledFunction();
            if (!callee)
                continue;
            size_t sz;
            if (callee == alloc_obj) {
                assert(call->getNumArgOperands() == 3);
                sz = (size_t)cast<ConstantInt>(call->getArgOperand(1))->getZExtValue();
            }
            else {
                continue;
            }
            allocs[call] = sz;
        }
    }

    auto &entry = F.getEntryBlock();
    auto first = &entry.front();
    for (auto it: allocs) {
        bool ignore_tag = true;
        auto orig = it.first;
        if (optimize && checkUses(orig, 0, ignore_tag)) {
            // The allocation does not escape or be used in a phi node so none of the derived
            // SSA from it are live when we run the allocation again.
            // It is now safe to promote the allocation to an entry block alloca.
            size_t sz = it.second;
            size_t align = 1;
            // TODO make codegen handling of alignment consistent and pass that as a parameter
            // to the allocation function directly.
            if (!ignore_tag) {
                align = sz <= 8 ? 8 : 16;
                sz += align;
            }
            else if (sz >= 16) {
                align = 16;
            }
            else if (sz >= 8) {
                align = 8;
            }
            else if (sz >= 4) {
                align = 4;
            }
            else if (sz >= 2) {
                align = 2;
            }
            const auto &dbg = orig->getDebugLoc();
#if JL_LLVM_VERSION >= 50000
            Instruction *ptr = new AllocaInst(T_int8, 0, ConstantInt::get(T_int32, sz),
                                              align, "", first);
#else
            Instruction *ptr = new AllocaInst(T_int8, ConstantInt::get(T_int32, sz),
                                              align, "", first);
#endif
            ptr->setDebugLoc(dbg);
            if (!ignore_tag) {
                ptr = GetElementPtrInst::CreateInBounds(T_size, ptr,
                                                        {ConstantInt::get(T_int32, align)}, "",
                                                        first);
                ptr->setDebugLoc(dbg);
            }
            auto cast = new BitCastInst(ptr, T_pjlvalue0, "", first);
            cast->setDebugLoc(dbg);
            // Someone might be reading the tag, initialize it.
            if (!ignore_tag) {
                auto tag = orig->getArgOperand(2);
                auto cast2 = new BitCastInst(ptr, T_ppjlvalue0, "", orig);
                cast2->setDebugLoc(dbg);
                auto tagaddr = GetElementPtrInst::Create(T_pjlvalue, cast,
                                                         {ConstantInt::get(T_size, -1)},
                                                         "", orig);
                tagaddr->setDebugLoc(dbg);
                auto store = new StoreInst(tag, tagaddr, orig);
                store->setMetadata(LLVMContext::MD_tbaa, tbaa_tag);
                store->setDebugLoc(dbg);
            }
            replaceUsesWith(orig, cast, ignore_tag);
        }
        else {
            lowerAlloc(orig, it.second);
        }
    }
    return true;
}

char AllocOpt::ID = 0;
static RegisterPass<AllocOpt> X("AllocOpt", "Promote heap allocation to stack",
                                false /* Only looks at CFG */,
                                false /* Analysis Pass */);

}

Pass *createAllocOptPass(bool opt)
{
    return new AllocOpt(opt);
}
