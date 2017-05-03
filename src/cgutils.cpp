// This file is a part of Julia. License is MIT: https://julialang.org/license

// utility procedures used in code generation

static Instruction *tbaa_decorate(MDNode *md, Instruction *load_or_store)
{
    load_or_store->setMetadata( llvm::LLVMContext::MD_tbaa, md );
    return load_or_store;
}

#define prepare_call(Callee) prepare_call_in(jl_Module, (Callee))
static Value *prepare_call_in(Module *M, Value *Callee)
{
    if (Function *F = dyn_cast<Function>(Callee)) {
        GlobalValue *local = M->getNamedValue(Callee->getName());
        if (!local) {
            local = function_proto(F, M);
        }
        return local;
    }
    return Callee;
}

static Value *maybe_decay_untracked(IRBuilder<> &irbuilder, Value *V)
{
    if (V->getType() == T_pjlvalue)
        return irbuilder.CreateAddrSpaceCast(V, T_prjlvalue);
    else if (V->getType() == T_ppjlvalue)
        return irbuilder.CreateBitCast(V, T_pprjlvalue);
    return V;
}

static Constant *maybe_decay_untracked(IRBuilder<> &irbuilder, Constant *C)
{
    if (C->getType() == T_pjlvalue)
        return ConstantExpr::getAddrSpaceCast(C, T_prjlvalue);
    else if (C->getType() == T_ppjlvalue)
        return ConstantExpr::getBitCast(C, T_pprjlvalue);
    return C;
}

static Value *decay_derived(IRBuilder<> &irbuilder, Value *V)
{
    Type *T = V->getType();
    if (cast<PointerType>(T)->getAddressSpace() == AddressSpace::Derived)
        return V;
    // Once llvm deletes pointer element types, we won't need it here any more either.
    Type *NewT = PointerType::get(cast<PointerType>(T)->getElementType(), AddressSpace::Derived);
    return irbuilder.CreateAddrSpaceCast(V, NewT);
}

static Value *mark_callee_rooted(IRBuilder<> &irbuilder, Value *V)
{
    assert(V->getType() == T_pjlvalue || V->getType() == T_prjlvalue);
    return irbuilder.CreateAddrSpaceCast(V,
        PointerType::get(T_jlvalue, AddressSpace::CalleeRooted));
}

#define maybe_decay_untracked(V)  maybe_decay_untracked(ctx.builder, (V))
#define maybe_decay_untracked(V)  maybe_decay_untracked(ctx.builder, (V))
#define decay_derived(V)          decay_derived(ctx.builder, (V))
#define mark_callee_rooted(V)     mark_callee_rooted(ctx.builder, (V))


// --- language feature checks ---

// branch on whether a language feature is enabled or not
#define JL_FEAT_TEST(ctx, feature) ((ctx).params->feature)

// require a language feature to be enabled
#define JL_FEAT_REQUIRE(ctx, feature) \
    if (!JL_FEAT_TEST(ctx, feature)) \
        jl_errorf("%s for %s:%d requires the " #feature " language feature, which is disabled", \
                  __FUNCTION__, (ctx).file.str().c_str(), *(ctx).line);


// --- hook checks ---

#define JL_HOOK_TEST(params,hook) ((params)->hooks.hook != jl_nothing)

#define JL_HOOK_CALL(params,hook,argc,...) \
    _hook_call<argc>((params)->hooks.hook, {{__VA_ARGS__}});
template<int N>
static inline void _hook_call(jl_value_t *hook, std::array<jl_value_t*,N> args) {
    jl_value_t **argv;
    JL_GC_PUSHARGS(argv, N+1);
    argv[0] = hook;
    for (int i = 0; i < N; i++)
        argv[i+1] = args[i];
    jl_apply(argv, N+1);
    JL_GC_POP();
}


// --- string constants ---
static StringMap<GlobalVariable*> stringConstants;
static Value *stringConstPtr(IRBuilder<> &irbuilder, const std::string &txt)
{
    StringRef ctxt(txt.c_str(), strlen(txt.c_str()) + 1);
    StringMap<GlobalVariable*>::iterator pooledval =
        stringConstants.insert(std::pair<StringRef, GlobalVariable*>(ctxt, NULL)).first;
    StringRef pooledtxt = pooledval->getKey();
    if (imaging_mode) {
        if (pooledval->second == NULL) {
            static int strno = 0;
            std::stringstream ssno;
            ssno << "_j_str" << strno++;
            GlobalVariable *gv = get_pointer_to_constant(
                                    ConstantDataArray::get(jl_LLVMContext,
                                                           ArrayRef<unsigned char>(
                                                           (const unsigned char*)pooledtxt.data(),
                                                           pooledtxt.size())),
                                    ssno.str(),
                                    *shadow_output);
            pooledval->second = gv;
            jl_ExecutionEngine->addGlobalMapping(gv, (void*)(uintptr_t)pooledtxt.data());
        }

        GlobalVariable *v = prepare_global_in(jl_builderModule(irbuilder), pooledval->second);
        Value *zero = ConstantInt::get(Type::getInt32Ty(jl_LLVMContext), 0);
        Value *Args[] = { zero, zero };
        return irbuilder.CreateInBoundsGEP(v->getValueType(), v, Args);
    }
    else {
        Value *v = ConstantExpr::getIntToPtr(
                ConstantInt::get(T_size, (uintptr_t)pooledtxt.data()),
                T_pint8);
        return v;
    }
}

// --- Debug info ---

static DIType *julia_type_to_di(jl_value_t *jt, DIBuilder *dbuilder, bool isboxed = false)
{
    if (isboxed)
        return jl_pvalue_dillvmt;
    // always return the boxed representation for types with hidden content
    if (jl_is_abstracttype(jt) || !jl_is_datatype(jt) || jl_is_array_type(jt) ||
        jt == (jl_value_t*)jl_sym_type || jt == (jl_value_t*)jl_module_type ||
        jt == (jl_value_t*)jl_simplevector_type || jt == (jl_value_t*)jl_datatype_type ||
        jt == (jl_value_t*)jl_method_instance_type)
        return jl_pvalue_dillvmt;
    if (jl_is_unionall(jt) || jl_is_typevar(jt))
        return jl_pvalue_dillvmt;
    assert(jl_is_datatype(jt));
    jl_datatype_t *jdt = (jl_datatype_t*)jt;
    if (jdt->ditype != NULL) {
        DIType* t = (DIType*)jdt->ditype;
        return t;
    }
    if (jl_is_primitivetype(jt)) {
        uint64_t SizeInBits = jl_datatype_nbits(jdt);
#if JL_LLVM_VERSION >= 40000
        llvm::DIType *t = dbuilder->createBasicType(
                jl_symbol_name(jdt->name->name),
                SizeInBits,
                llvm::dwarf::DW_ATE_unsigned);
        jdt->ditype = t;
        return t;
#else
        llvm::DIType *t = dbuilder->createBasicType(
                jl_symbol_name(jdt->name->name),
                SizeInBits,
                8 * jl_datatype_align(jdt),
                llvm::dwarf::DW_ATE_unsigned);
        jdt->ditype = t;
        return t;
#endif
    }
    else if (!jl_is_leaf_type(jt)) {
        jdt->ditype = jl_pvalue_dillvmt;
        return jl_pvalue_dillvmt;
    }
    else if (jl_is_structtype(jt)) {
        jl_datatype_t *jst = (jl_datatype_t*)jt;
        size_t ntypes = jl_datatype_nfields(jst);
        const char *tname = jl_symbol_name(jdt->name->name);
        std::stringstream unique_name;
        unique_name << tname << "_" << globalUnique++;
        llvm::DICompositeType *ct = dbuilder->createStructType(
            NULL,                       // Scope
            tname,                      // Name
            NULL,                       // File
            0,                          // LineNumber
            jl_datatype_nbits(jdt),     // SizeInBits
            8 * jl_datatype_align(jdt), // AlignInBits
            DIFlagZero,                 // Flags
            NULL,                       // DerivedFrom
            DINodeArray(),              // Elements
            dwarf::DW_LANG_Julia,       // RuntimeLanguage
            nullptr,                    // VTableHolder
            unique_name.str()           // UniqueIdentifier
            );
        jdt->ditype = ct;
        std::vector<llvm::Metadata*> Elements;
        for(unsigned i = 0; i < ntypes; i++)
            Elements.push_back(julia_type_to_di(jl_svecref(jst->types,i),dbuilder,false));
        dbuilder->replaceArrays(ct, dbuilder->getOrCreateArray(ArrayRef<Metadata*>(Elements)));
        return ct;
    }
    else {
        assert(jl_is_datatype(jt));
        jdt->ditype = dbuilder->createTypedef(jl_pvalue_dillvmt,
            jl_symbol_name(jdt->name->name), NULL, 0, NULL);
        return (llvm::DIType*)jdt->ditype;
    }
    // TODO: Fixme
    return jl_pvalue_dillvmt;
}

static Value *emit_pointer_from_objref(jl_codectx_t &ctx, Value *V)
{
    unsigned AS = cast<PointerType>(V->getType())->getAddressSpace();
    if (AS != AddressSpace::Tracked && AS != AddressSpace::Derived)
        return ctx.builder.CreateBitCast(V, T_pjlvalue);
    V = ctx.builder.CreateBitCast(decay_derived(V),
            PointerType::get(T_jlvalue, AddressSpace::Derived));
    CallInst *Call = ctx.builder.CreateCall(prepare_call(pointer_from_objref_func), V);
#if JL_LLVM_VERSION >= 50000
    Call->addAttribute(AttributeList::FunctionIndex, Attribute::ReadNone);
#else
    Call->addAttribute(AttributeSet::FunctionIndex, Attribute::ReadNone);
#endif
    return Call;
}

// --- emitting pointers directly into code ---

static Constant *literal_static_pointer_val(jl_codectx_t &ctx, const void *p, Type *t)
{
    // this function will emit a static pointer into the generated code
    // the generated code will only be valid during the current session,
    // and thus, this should typically be avoided in new API's
#if defined(_P64)
    return ConstantExpr::getPointerBitCastOrAddrSpaceCast(ConstantExpr::getIntToPtr(ConstantInt::get(T_int64, (uint64_t)p), T_pjlvalue), t);
#else
    return ConstantExpr::getPointerBitCastOrAddrSpaceCast(ConstantExpr::getIntToPtr(ConstantInt::get(T_int32, (uint32_t)p), T_pjlvalue), t);
#endif
}


static Value *julia_pgv(jl_codectx_t &ctx, const char *cname, void *addr)
{
    // emit a GlobalVariable for a jl_value_t named "cname"
    return jl_get_global_for(cname, addr, jl_Module);
}

static Value *julia_pgv(jl_codectx_t &ctx, const char *prefix, jl_sym_t *name, jl_module_t *mod, void *addr)
{
    // emit a GlobalVariable for a jl_value_t, using the prefix, name, and module to
    // to create a readable name of the form prefixModA.ModB.name
    size_t len = strlen(jl_symbol_name(name))+strlen(prefix)+1;
    jl_module_t *parent = mod, *prev = NULL;
    while (parent != NULL && parent != prev) {
        len += strlen(jl_symbol_name(parent->name))+1;
        prev = parent;
        parent = parent->parent;
    }
    char *fullname = (char*)alloca(len);
    strcpy(fullname, prefix);
    int skipfirst = jl_symbol_name(name)[0] == '@';
    len -= strlen(jl_symbol_name(name)) + 1 - skipfirst;
    strcpy(fullname + len, jl_symbol_name(name) + skipfirst);
    parent = mod;
    prev = NULL;
    while (parent != NULL && parent != prev) {
        size_t part = strlen(jl_symbol_name(parent->name))+1-skipfirst;
        strcpy(fullname+len-part,jl_symbol_name(parent->name)+skipfirst);
        fullname[len-1] = '.';
        len -= part;
        prev = parent;
        parent = parent->parent;
    }
    return julia_pgv(ctx, fullname, addr);
}

static GlobalVariable *julia_const_gv(jl_value_t *val);
static Value *literal_pointer_val_slot(jl_codectx_t &ctx, jl_value_t *p)
{
    // emit a pointer to a jl_value_t* which will allow it to be valid across reloading code
    // also, try to give it a nice name for gdb, for easy identification
    if (!imaging_mode) {
        Module *M = jl_Module;
        GlobalVariable *gv = new GlobalVariable(
                *M, T_pjlvalue, true, GlobalVariable::PrivateLinkage,
                literal_static_pointer_val(ctx, p, T_pjlvalue));
        gv->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
        return gv;
    }
    if (GlobalVariable *gv = julia_const_gv(p)) {
        // if this is a known object, use the existing GlobalValue
        return maybe_decay_untracked(prepare_global(gv));
    }
    if (jl_is_datatype(p)) {
        jl_datatype_t *addr = (jl_datatype_t*)p;
        // DataTypes are prefixed with a +
        return maybe_decay_untracked(julia_pgv(ctx, "+", addr->name->name, addr->name->module, p));
    }
    if (jl_is_method(p)) {
        jl_method_t *m = (jl_method_t*)p;
        // functions are prefixed with a -
        return maybe_decay_untracked(julia_pgv(ctx, "-", m->name, m->module, p));
    }
    if (jl_is_method_instance(p)) {
        jl_method_instance_t *linfo = (jl_method_instance_t*)p;
        // Type-inferred functions are also prefixed with a -
        if (jl_is_method(linfo->def.method))
            return maybe_decay_untracked(julia_pgv(ctx, "-", linfo->def.method->name, linfo->def.method->module, p));
    }
    if (jl_is_symbol(p)) {
        jl_sym_t *addr = (jl_sym_t*)p;
        // Symbols are prefixed with jl_sym#
        return maybe_decay_untracked(julia_pgv(ctx, "jl_sym#", addr, NULL, p));
    }
    // something else gets just a generic name
    return maybe_decay_untracked(julia_pgv(ctx, "jl_global#", p));
}

static Value *literal_pointer_val(jl_codectx_t &ctx, jl_value_t *p)
{
    if (p == NULL)
        return V_null;
    if (!imaging_mode)
        return literal_static_pointer_val(ctx, p, T_prjlvalue);
    Value *pgv = literal_pointer_val_slot(ctx, p);
    return tbaa_decorate(tbaa_const, ctx.builder.CreateLoad(pgv));
}

static Value *literal_pointer_val(jl_codectx_t &ctx, jl_binding_t *p)
{
    // emit a pointer to any jl_value_t which will be valid across reloading code
    if (p == NULL)
        return V_null;
    if (!imaging_mode)
        return literal_static_pointer_val(ctx, p, T_pjlvalue);
    // bindings are prefixed with jl_bnd#
    Value *pgv = julia_pgv(ctx, "jl_bnd#", p->name, p->owner, p);
    return tbaa_decorate(tbaa_const, ctx.builder.CreateLoad(pgv));
}

// bitcast a value, but preserve its address space when dealing with pointer types
static Value *emit_bitcast(jl_codectx_t &ctx, Value *v, Type *jl_value)
{
    if (isa<PointerType>(jl_value) &&
        v->getType()->getPointerAddressSpace() != jl_value->getPointerAddressSpace()) {
        // Cast to the proper address space
        Type *jl_value_addr =
                PointerType::get(cast<PointerType>(jl_value)->getElementType(),
                                 v->getType()->getPointerAddressSpace());
        return ctx.builder.CreateBitCast(v, jl_value_addr);
    }
    else {
        return ctx.builder.CreateBitCast(v, jl_value);
    }
}

static Value *julia_binding_gv(jl_codectx_t &ctx, Value *bv)
{
    Value *offset = ConstantInt::get(T_size, offsetof(jl_binding_t, value) / sizeof(size_t));
    return ctx.builder.CreateGEP(bv, offset);
}

static Value *julia_binding_gv(jl_codectx_t &ctx, jl_binding_t *b)
{
    // emit a literal_pointer_val to the value field of a jl_binding_t
    // binding->value are prefixed with *
    Value *bv;
    if (imaging_mode)
        bv = emit_bitcast(ctx,
                tbaa_decorate(tbaa_const,
                              ctx.builder.CreateLoad(julia_pgv(ctx, "*", b->name, b->owner, b))),
                T_pprjlvalue);
    else
        bv = literal_static_pointer_val(ctx, b, T_pprjlvalue);
    return julia_binding_gv(ctx, bv);
}

// --- mapping between julia and llvm types ---

static Type *julia_struct_to_llvm(jl_value_t *jt, jl_unionall_t *ua_env, bool *isboxed);

extern "C" {
JL_DLLEXPORT Type *julia_type_to_llvm(jl_value_t *jt, bool *isboxed)
{
    // this function converts a Julia Type into the equivalent LLVM type
    if (isboxed) *isboxed = false;
    if (jt == (jl_value_t*)jl_bottom_type)
        return T_void;
    if (jl_is_leaf_type(jt)) {
        if ((jl_is_primitivetype(jt) || jl_isbits(jt))) {
            if (jl_datatype_nbits(jt) == 0)
                return T_void;
            Type *t = julia_struct_to_llvm(jt, NULL, isboxed);
            assert(t != NULL);
            return t;
        }
    }
    if (isboxed) *isboxed = true;
    return T_pjlvalue;
}
}

// converts a julia bitstype into the equivalent LLVM bitstype
static Type *bitstype_to_llvm(jl_value_t *bt)
{
    assert(jl_is_primitivetype(bt));
    if (bt == (jl_value_t*)jl_bool_type)
        return T_int8;
    if (bt == (jl_value_t*)jl_long_type)
        return T_size;
    if (bt == (jl_value_t*)jl_float32_type)
        return T_float32;
    if (bt == (jl_value_t*)jl_float64_type)
        return T_float64;
    if (jl_is_cpointer_type(bt)) {
        Type *lt = julia_type_to_llvm(jl_tparam0(bt));
        if (lt == T_void)
            return T_pint8;
        return PointerType::get(lt, 0);
    }
    int nb = jl_datatype_size(bt);
    return Type::getIntNTy(jl_LLVMContext, nb * 8);
}

// compute whether all leaf subtypes of this type have the same layout
// (which is conservatively approximated here by asking whether the types of any of the
// fields depend on any of the parameters of the containing type)
static bool julia_struct_has_layout(jl_datatype_t *dt, jl_unionall_t *ua)
{
    if (dt->layout || dt->struct_decl || jl_is_primitivetype(dt) || jl_isbits(dt))
        return true;
    if (ua) {
        size_t i, ntypes = jl_svec_len(dt->types);
        for (i = 0; i < ntypes; i++) {
            jl_value_t *ty = jl_svecref(dt->types, i);
            if (jl_has_typevar_from_unionall(ty, ua))
                return false;
        }
    }
    return true;
}

static Type *julia_struct_to_llvm(jl_value_t *jt, jl_unionall_t *ua, bool *isboxed)
{
    // this function converts a Julia Type into the equivalent LLVM struct
    // use this where C-compatible (unboxed) structs are desired
    // use julia_type_to_llvm directly when you want to preserve Julia's type semantics
    if (isboxed) *isboxed = false;
    if (jt == (jl_value_t*)jl_bottom_type)
        return T_void;
    if (jl_is_primitivetype(jt))
        return bitstype_to_llvm(jt);
    bool isTuple = jl_is_tuple_type(jt);
    if ((isTuple || jl_is_structtype(jt)) && !jl_is_array_type(jt)) {
        jl_datatype_t *jst = (jl_datatype_t*)jt;
        if (jst->struct_decl == NULL) {
            size_t i, ntypes = jl_svec_len(jst->types);
            if (ntypes == 0 || (jst->layout && jl_datatype_nbits(jst) == 0))
                return T_void;
            if (!julia_struct_has_layout(jst, ua))
                return NULL;
            StructType *structdecl;
            if (!isTuple) {
                structdecl = StructType::create(jl_LLVMContext, jl_symbol_name(jst->name->name));
                jst->struct_decl = structdecl;
            }
            std::vector<Type*> latypes(0);
            bool isarray = true;
            bool isvector = true;
            jl_value_t *jlasttype = NULL;
            Type *lasttype = NULL;
            bool allghost = true;
            for (i = 0; i < ntypes; i++) {
                jl_value_t *ty = jl_svecref(jst->types, i);
                if (jlasttype != NULL && ty != jlasttype)
                    isvector = false;
                jlasttype = ty;
                bool isptr;
                if (jst->layout)
                    isptr = jl_field_isptr(jst, i);
                else // compute what jl_compute_field_offsets would say
                    isptr = jl_isbits(ty) && jl_is_leaf_type(ty) && ((jl_datatype_t*)ty)->layout;
                Type *lty;
                if (isptr)
                    lty = T_pjlvalue;
                else if (ty == (jl_value_t*)jl_bool_type)
                    lty = T_int8;
                else
                    lty = julia_type_to_llvm(ty);
                if (lasttype != NULL && lasttype != lty)
                    isarray = false;
                lasttype = lty;
                if (type_is_ghost(lty))
                    lty = NoopType;
                else
                    allghost = false;
                latypes.push_back(lty);
            }
            if (allghost) {
                assert(jst->layout == NULL); // otherwise should have been caught above
                jst->struct_decl = T_void;
            }
            else if (!isTuple) {
                if (jl_is_vecelement_type(jt))
                    // VecElement type is unwrapped in LLVM
                    jst->struct_decl = latypes[0];
                else
                    structdecl->setBody(latypes);
            }
            else {
                if (isarray && lasttype != T_int1 && !type_is_ghost(lasttype)) {
                    if (isvector && jl_special_vector_alignment(ntypes, jlasttype) != 0)
                        jst->struct_decl = VectorType::get(lasttype, ntypes);
                    else
                        jst->struct_decl = ArrayType::get(lasttype, ntypes);
                }
                else {
                    jst->struct_decl = StructType::get(jl_LLVMContext, ArrayRef<Type*>(&latypes[0], ntypes));
                }
            }
#ifndef JL_NDEBUG
            // If LLVM and Julia disagree about alignment, much trouble ensues, so check it!
            if (jst->layout) {
                const DataLayout &DL =
#if JL_LLVM_VERSION >= 40000
                    jl_data_layout;
#else
                    jl_ExecutionEngine->getDataLayout();
#endif
                unsigned llvm_alignment = DL.getABITypeAlignment((Type*)jst->struct_decl);
                unsigned julia_alignment = jl_datatype_align(jst);
                // Check that the alignment adheres to the heap alignment.
                assert(julia_alignment <= JL_HEAP_ALIGNMENT);
                // TODO: Fix alignment calculation in LLVM, as well as in the GC and the struct declaration
                if (llvm_alignment  <= JL_HEAP_ALIGNMENT)
                    assert(julia_alignment == llvm_alignment);
            }
#endif
        }
        return (Type*)jst->struct_decl;
    }
    if (isboxed) *isboxed = true;
    return T_pjlvalue;
}

static bool is_datatype_all_pointers(jl_datatype_t *dt)
{
    size_t i, l = jl_datatype_nfields(dt);
    for(i=0; i < l; i++) {
        if (!jl_field_isptr(dt, i)) {
            return false;
        }
    }
    return true;
}

static bool is_tupletype_homogeneous(jl_svec_t *t, bool allow_va = false)
{
    size_t i, l = jl_svec_len(t);
    if (l > 0) {
        jl_value_t *t0 = jl_svecref(t, 0);
        if (!jl_is_leaf_type(t0)) {
            if (allow_va && jl_is_vararg_type(t0) &&
                  jl_is_leaf_type(jl_unwrap_vararg(t0)))
                return true;
            return false;
        }
        for(i=1; i < l; i++) {
            if (allow_va && i == l - 1 && jl_is_vararg_type(jl_svecref(t,i))) {
                if (t0 != jl_unwrap_vararg(jl_svecref(t,i)))
                    return false;
                continue;
            }
            if (t0 != jl_svecref(t,i))
                return false;
        }
    }
    return true;
}

static bool deserves_sret(jl_value_t *dt, Type *T)
{
    assert(jl_is_datatype(dt));
    return (size_t)jl_datatype_size(dt) > sizeof(void*) && !T->isFloatingPointTy() && !T->isVectorTy();
}

static bool for_each_uniontype_small(
        std::function<void(unsigned, jl_datatype_t*)> f,
        jl_value_t *ty,
        unsigned &counter)
{
    if (counter > 127)
        return false;
    if (jl_is_uniontype(ty)) {
        bool allunbox = for_each_uniontype_small(f, ((jl_uniontype_t*)ty)->a, counter);
        allunbox &= for_each_uniontype_small(f, ((jl_uniontype_t*)ty)->b, counter);
        return allunbox;
    }
    else if (isbits_spec(ty)) {
        f(++counter, (jl_datatype_t*)ty);
        return true;
    }
    return false;
}

static Value *emit_typeof_boxed(jl_codectx_t &ctx, const jl_cgval_t &p);

static unsigned get_box_tindex(jl_datatype_t *jt, jl_value_t *ut)
{
    unsigned new_idx = 0;
    unsigned new_counter = 0;
    for_each_uniontype_small(
            // find the corresponding index in the new union-type
            [&](unsigned new_idx_, jl_datatype_t *new_jt) {
                if (jt == new_jt)
                    new_idx = new_idx_;
            },
            ut,
            new_counter);
    return new_idx;
}


// --- generating various field accessors ---

static Value *emit_nthptr_addr(jl_codectx_t &ctx, Value *v, ssize_t n, bool gctracked = true)
{
    return ctx.builder.CreateGEP(emit_bitcast(ctx, gctracked ? decay_derived(v) : v, T_pprjlvalue),
                             ConstantInt::get(T_size, n));
}

static Value *emit_nthptr_addr(jl_codectx_t &ctx, Value *v, Value *idx, bool gctracked = true)
{
    return ctx.builder.CreateGEP(emit_bitcast(ctx, gctracked ? decay_derived(v) : v, T_pprjlvalue), idx);
}

static Value *emit_nthptr(jl_codectx_t &ctx, Value *v, ssize_t n, MDNode *tbaa)
{
    // p = (jl_value_t**)v; p[n]
    Value *vptr = emit_nthptr_addr(ctx, v, n);
    return tbaa_decorate(tbaa, ctx.builder.CreateLoad(vptr, false));
}

static Value *emit_nthptr_recast(jl_codectx_t &ctx, Value *v, Value *idx, MDNode *tbaa, Type *ptype, bool gctracked = true)
{
    // p = (jl_value_t**)v; *(ptype)&p[n]
    Value *vptr = emit_nthptr_addr(ctx, v, idx, gctracked);
    return tbaa_decorate(tbaa, ctx.builder.CreateLoad(emit_bitcast(ctx, vptr, ptype), false));
}

static Value *emit_nthptr_recast(jl_codectx_t &ctx, Value *v, ssize_t n, MDNode *tbaa, Type *ptype, bool gctracked = true)
{
    // p = (jl_value_t**)v; *(ptype)&p[n]
    Value *vptr = emit_nthptr_addr(ctx, v, n, gctracked);
    return tbaa_decorate(tbaa, ctx.builder.CreateLoad(emit_bitcast(ctx, vptr, ptype), false));
}

static Value *emit_typeptr_addr(jl_codectx_t &ctx, Value *p)
{
    ssize_t offset = (sizeof(jl_taggedvalue_t) -
                      offsetof(jl_taggedvalue_t, type)) / sizeof(jl_value_t*);
    return emit_nthptr_addr(ctx, p, -offset);
}

static Value *boxed(jl_codectx_t &ctx, const jl_cgval_t &v, bool gcooted=true);
static Value *boxed(jl_codectx_t &ctx, const jl_cgval_t &v, jl_value_t* type) = delete; // C++11 (temporary to prevent rebase error)

static Value* mask_gc_bits(jl_codectx_t &ctx, Value *tag)
{
    return ctx.builder.CreateIntToPtr(ctx.builder.CreateAnd(
                ctx.builder.CreatePtrToInt(tag, T_size),
                ConstantInt::get(T_size, ~(uintptr_t)15)),
            tag->getType());
}

static Value *emit_typeof(jl_codectx_t &ctx, Value *tt)
{
    assert(tt != NULL && !isa<AllocaInst>(tt) && "expected a conditionally boxed value");
    // given p, a jl_value_t*, compute its type tag
    // The raw address is not GC-safe to load from as it may have mask bits set.
    // Note that this gives the optimizer license to not root this value. That
    // is fine however, since leaf types are not GCed at the moment. Should
    // that ever change, this may have to go through a special intrinsic.
    Value *addr = emit_bitcast(ctx, emit_typeptr_addr(ctx, tt), T_ppjlvalue);
    tt = tbaa_decorate(tbaa_tag, ctx.builder.CreateLoad(addr));
    return maybe_decay_untracked(mask_gc_bits(ctx, tt));
}

static jl_cgval_t emit_typeof(jl_codectx_t &ctx, const jl_cgval_t &p)
{
    // given p, compute its type
    if (p.constant)
        return mark_julia_const(jl_typeof(p.constant));
    if (p.isboxed && !jl_is_leaf_type(p.typ)) {
        return mark_julia_type(ctx, emit_typeof(ctx, p.V), true, jl_datatype_type, /*needsroot*/false);
    }
    if (p.TIndex) {
        Value *tindex = ctx.builder.CreateAnd(p.TIndex, ConstantInt::get(T_int8, 0x7f));
        Value *pdatatype;
        unsigned counter;
        counter = 0;
        bool allunboxed = for_each_uniontype_small(
                [&](unsigned idx, jl_datatype_t *jt) { },
                p.typ,
                counter);
        if (allunboxed)
            pdatatype = Constant::getNullValue(T_ppjlvalue);
        else {
            // See note above in emit_typeof(Value*), we can't tell the system
            // about this until we've cleared the GC bits.
            pdatatype = emit_bitcast(ctx, emit_typeptr_addr(ctx, ctx.builder.CreateLoad(p.gcroot)), T_ppjlvalue);
        }
        counter = 0;
        for_each_uniontype_small(
                [&](unsigned idx, jl_datatype_t *jt) {
                    Value *cmp = ctx.builder.CreateICmpEQ(tindex, ConstantInt::get(T_int8, idx));
                    pdatatype = ctx.builder.CreateSelect(cmp,
                        decay_derived(emit_bitcast(ctx, literal_pointer_val_slot(ctx, (jl_value_t*)jt), T_ppjlvalue)),
                        decay_derived(pdatatype));
                },
                p.typ,
                counter);
        Value *datatype;
        if (allunboxed) {
            datatype = tbaa_decorate(tbaa_const, ctx.builder.CreateLoad(maybe_decay_untracked(pdatatype)));
        }
        else {
            datatype = maybe_decay_untracked(mask_gc_bits(ctx, tbaa_decorate(tbaa_tag, ctx.builder.CreateLoad(pdatatype))));
        }
        return mark_julia_type(ctx, datatype, true, jl_datatype_type, /*needsroot*/false);
    }
    jl_value_t *aty = p.typ;
    if (jl_is_type_type(aty)) {
        // convert Int::Type{Int} ==> typeof(Int) ==> DataType
        // but convert 1::Type{1} ==> typeof(1) ==> Int
        aty = (jl_value_t*)jl_typeof(jl_tparam0(aty));
    }
    return mark_julia_const(aty);
}

static Value *emit_typeof_boxed(jl_codectx_t &ctx, const jl_cgval_t &p)
{
    return boxed(ctx, emit_typeof(ctx, p));
}

static Value *emit_datatype_types(jl_codectx_t &ctx, Value *dt)
{
    Value *Ptr = emit_bitcast(ctx, decay_derived(dt), T_ppjlvalue);
    Value *Idx = ConstantInt::get(T_size, offsetof(jl_datatype_t, types) / sizeof(void*));
    return tbaa_decorate(tbaa_const, ctx.builder.CreateLoad(T_pjlvalue, ctx.builder.CreateGEP(T_pjlvalue, Ptr, Idx)));
}

static Value *emit_datatype_nfields(jl_codectx_t &ctx, Value *dt)
{
    Value *type_svec = emit_bitcast(ctx, emit_datatype_types(ctx, dt), T_psize);
    return tbaa_decorate(tbaa_const, ctx.builder.CreateLoad(T_size, type_svec));
}

static Value *emit_datatype_size(jl_codectx_t &ctx, Value *dt)
{
    Value *Ptr = emit_bitcast(ctx, decay_derived(dt), T_pint32);
    Value *Idx = ConstantInt::get(T_size, offsetof(jl_datatype_t, size) / sizeof(int));
    return tbaa_decorate(tbaa_const, ctx.builder.CreateLoad(T_int32, ctx.builder.CreateGEP(T_int32, Ptr, Idx)));
}

/* this is valid code, it's simply unused
static Value *emit_sizeof(jl_codectx_t &ctx, const jl_cgval_t &p)
{
    if (p.TIndex) {
        Value *tindex = ctx.builder.CreateAnd(p.TIndex, ConstantInt::get(T_int8, 0x7f));
        Value *size = ConstantInt::get(T_int32, -1);
        unsigned counter = 0;
        bool allunboxed = for_each_uniontype_small(
                [&](unsigned idx, jl_datatype_t *jt) {
                    Value *cmp = ctx.builder.CreateICmpEQ(tindex, ConstantInt::get(T_int8, idx));
                    size = ctx.builder.CreateSelect(cmp, ConstantInt::get(T_int32, jl_datatype_size(jt)), size);
                },
                p.typ,
                counter);
        if (!allunboxed && p.ispointer() && p.V && !isa<AllocaInst>(p.V)) {
            BasicBlock *currBB = ctx.builder.GetInsertBlock();
            BasicBlock *dynloadBB = BasicBlock::Create(jl_LLVMContext, "dyn_sizeof", ctx.f);
            BasicBlock *postBB = BasicBlock::Create(jl_LLVMContext, "post_sizeof", ctx.f);
            Value *isboxed = ctx.builder.CreateICmpNE(
                    ctx.builder.CreateAnd(p.TIndex, ConstantInt::get(T_int8, 0x80)),
                    ConstantInt::get(T_int8, 0));
            ctx.builder.CreateCondBr(isboxed, dynloadBB, postBB);
            ctx.builder.SetInsertPoint(dynloadBB);
            Value *datatype = emit_typeof(p.V);
            Value *dyn_size = emit_datatype_size(ctx, datatype);
            ctx.builder.CreateBr(postBB);
            ctx.builder.SetInsertPoint(postBB);
            PHINode *sizeof_merge = ctx.builder.CreatePHI(T_int32, 2);
            sizeof_merge->addIncoming(dyn_size, dynloadBB);
            sizeof_merge->addIncoming(size, currBB);
            size = sizeof_merge;
        }
#ifndef NDEBUG
        // try to catch codegen errors early, before it uses this to memcpy over the entire stack
        CreateConditionalAbort(ctx.builder, ctx.builder.CreateICmpEQ(size, ConstantInt::get(T_int32, -1)));
#endif
        return size;
    }
    else if (jl_is_leaf_type(p.typ)) {
        return ConstantInt::get(T_int32, jl_datatype_size(p.typ));
    }
    else {
        Value *datatype = emit_typeof_boxed(ctx, p);
        Value *dyn_size = emit_datatype_size(ctx, datatype);
        return dyn_size;
    }
}
*/

static Value *emit_datatype_mutabl(jl_codectx_t &ctx, Value *dt)
{
    Value *Ptr = emit_bitcast(ctx, decay_derived(dt), T_pint8);
    Value *Idx = ConstantInt::get(T_size, offsetof(jl_datatype_t, mutabl));
    Value *mutabl = tbaa_decorate(tbaa_const,
            ctx.builder.CreateLoad(T_int8, ctx.builder.CreateGEP(T_int8, Ptr, Idx)));
    return ctx.builder.CreateTrunc(mutabl, T_int1);
}

static Value *emit_datatype_abstract(jl_codectx_t &ctx, Value *dt)
{
    Value *Ptr = emit_bitcast(ctx, decay_derived(dt), T_pint8);
    Value *Idx = ConstantInt::get(T_size, offsetof(jl_datatype_t, abstract));

    Value *abstract = tbaa_decorate(tbaa_const,
            ctx.builder.CreateLoad(T_int8, ctx.builder.CreateGEP(T_int8, Ptr, Idx)));
    return ctx.builder.CreateTrunc(abstract, T_int1);
}

static Value *emit_datatype_isbitstype(jl_codectx_t &ctx, Value *dt)
{
    Value *immut = ctx.builder.CreateXor(emit_datatype_mutabl(ctx, dt), ConstantInt::get(T_int1, -1));
    Value *nofields = ctx.builder.CreateICmpEQ(emit_datatype_nfields(ctx, dt), ConstantInt::get(T_size, 0));
    Value *isbitstype = ctx.builder.CreateAnd(immut, ctx.builder.CreateAnd(nofields,
            ctx.builder.CreateXor(ctx.builder.CreateAnd(emit_datatype_abstract(ctx, dt),
                    ctx.builder.CreateICmpSGT(emit_datatype_size(ctx, dt), ConstantInt::get(T_int32, 0))),
                ConstantInt::get(T_int1, -1))));
    return isbitstype;
}

static Value *emit_datatype_name(jl_codectx_t &ctx, Value *dt)
{
    return emit_nthptr(
            ctx, dt,
            (ssize_t)(offsetof(jl_datatype_t, name) / sizeof(char*)),
            tbaa_const);
}

// --- generating various error checks ---
// Do not use conditional throw for cases that type inference can know
// the error is always thrown. This may cause non dominated use
// of SSA value error in the verifier.

static void just_emit_error(jl_codectx_t &ctx, const std::string &txt)
{
    ctx.builder.CreateCall(prepare_call(jlerror_func), stringConstPtr(ctx.builder, txt));
}

static void emit_error(jl_codectx_t &ctx, const std::string &txt)
{
    just_emit_error(ctx, txt);
    ctx.builder.CreateUnreachable();
    BasicBlock *cont = BasicBlock::Create(jl_LLVMContext,"after_error",ctx.f);
    ctx.builder.SetInsertPoint(cont);
}

// DO NOT PASS IN A CONST CONDITION!
static void error_unless(jl_codectx_t &ctx, Value *cond, const std::string &msg)
{
    BasicBlock *failBB = BasicBlock::Create(jl_LLVMContext,"fail",ctx.f);
    BasicBlock *passBB = BasicBlock::Create(jl_LLVMContext,"pass");
    ctx.builder.CreateCondBr(cond, passBB, failBB);
    ctx.builder.SetInsertPoint(failBB);
    just_emit_error(ctx, msg);
    ctx.builder.CreateUnreachable();
    ctx.f->getBasicBlockList().push_back(passBB);
    ctx.builder.SetInsertPoint(passBB);
}

static void raise_exception(jl_codectx_t &ctx, Value *exc,
                            BasicBlock *contBB=nullptr)
{
    if (JL_HOOK_TEST(ctx.params, raise_exception)) {
        JL_HOOK_CALL(ctx.params, raise_exception, 2,
                     jl_box_voidpointer(wrap(ctx.builder.GetInsertBlock())),
                     jl_box_voidpointer(wrap(exc)));
    } else {
        JL_FEAT_REQUIRE(ctx, runtime);
        ctx.builder.CreateCall(prepare_call(jlthrow_func), { mark_callee_rooted(exc) });
    }
    ctx.builder.CreateUnreachable();
    if (!contBB) {
        contBB = BasicBlock::Create(jl_LLVMContext, "after_throw", ctx.f);
    }
    else {
        ctx.f->getBasicBlockList().push_back(contBB);
    }
    ctx.builder.SetInsertPoint(contBB);
}

// DO NOT PASS IN A CONST CONDITION!
static void raise_exception_unless(jl_codectx_t &ctx, Value *cond, Value *exc)
{
    BasicBlock *failBB = BasicBlock::Create(jl_LLVMContext,"fail",ctx.f);
    BasicBlock *passBB = BasicBlock::Create(jl_LLVMContext,"pass");
    ctx.builder.CreateCondBr(cond, passBB, failBB);
    ctx.builder.SetInsertPoint(failBB);
    raise_exception(ctx, exc, passBB);
}

// DO NOT PASS IN A CONST CONDITION!
static void raise_exception_if(jl_codectx_t &ctx, Value *cond, Value *exc)
{
    raise_exception_unless(ctx, ctx.builder.CreateXor(cond, ConstantInt::get(T_int1,-1)),
                           exc);
}

static size_t dereferenceable_size(jl_value_t *jt) {
    size_t size = 0;
    if (jl_is_array_type(jt)) {
        // Array has at least this much data
        size = sizeof(jl_array_t);
    } else {
        size = jl_datatype_size(jt);
    }
    return size;
}

static inline void maybe_mark_argument_dereferenceable(Argument *A, jl_value_t *jt) {
    if (!jl_is_leaf_type(jt)) {
        return;
    }
    size_t size = dereferenceable_size(jt);
    if (!size) {
        return;
    }
    A->getParent()->addDereferenceableAttr(A->getArgNo() + 1, size);
}

static inline Instruction *maybe_mark_load_dereferenceable(Instruction *LI, bool can_be_null, size_t size) {
    if (!size) {
        return LI;
    }
    llvm::SmallVector<Metadata *, 1> OPs;
    OPs.push_back(ConstantAsMetadata::get(ConstantInt::get(T_int64, size)));
    LI->setMetadata(can_be_null ? "dereferenceable_or_null" :
                                  "dereferenceable",
                    MDNode::get(jl_LLVMContext, OPs));
    return LI;
}

static inline Instruction *maybe_mark_load_dereferenceable(Instruction *LI, bool can_be_null, jl_value_t *jt) {
    if (!jl_is_leaf_type(jt)) {
        return LI;
    }
    size_t size = dereferenceable_size(jt);
    return maybe_mark_load_dereferenceable(LI, can_be_null, size);
}

static void null_pointer_check(jl_codectx_t &ctx, Value *v)
{
    raise_exception_unless(ctx,
            ctx.builder.CreateICmpNE(v, Constant::getNullValue(v->getType())),
            literal_pointer_val(ctx, jl_undefref_exception));
}

static void emit_type_error(jl_codectx_t &ctx, const jl_cgval_t &x, Value *type, const std::string &msg)
{
    Value *fname_val = stringConstPtr(ctx.builder, ctx.funcName);
    Value *msg_val = stringConstPtr(ctx.builder, msg);
    ctx.builder.CreateCall(prepare_call(jltypeerror_func),
                       { fname_val, msg_val,
                         type, mark_callee_rooted(boxed(ctx, x, false))});
}

static std::pair<Value*, bool> emit_isa(jl_codectx_t &ctx, const jl_cgval_t &x, jl_value_t *type, const std::string *msg)
{
    Optional<bool> known_isa;
    if (x.constant)
        known_isa = jl_isa(x.constant, type);
    else if (jl_subtype(x.typ, type))
        known_isa = true;
    else if (jl_type_intersection(x.typ, type) == (jl_value_t*)jl_bottom_type)
        known_isa = false;
    if (known_isa) {
        if (!*known_isa && msg) {
            emit_type_error(ctx, x, literal_pointer_val(ctx, type), *msg);
            ctx.builder.CreateUnreachable();
            BasicBlock *failBB = BasicBlock::Create(jl_LLVMContext, "fail", ctx.f);
            ctx.builder.SetInsertPoint(failBB);
        }
        return std::make_pair(ConstantInt::get(T_int1, *known_isa), true);
    }

    // intersection with Type needs to be handled specially
    if (jl_has_intersect_type_not_kind(type)) {
        Value *vx = maybe_decay_untracked(boxed(ctx, x));
        Value *vtyp = literal_pointer_val(ctx, type);
        if (msg && *msg == "typeassert") {
            ctx.builder.CreateCall(prepare_call(jltypeassert_func), { vx, vtyp });
            return std::make_pair(ConstantInt::get(T_int1, 1), true);
        }
        return std::make_pair(ctx.builder.CreateICmpNE(
                ctx.builder.CreateCall(prepare_call(jlisa_func), { vx, vtyp }),
                ConstantInt::get(T_int32, 0)), false);
    }
    // tests for isa leaftype can be handled with pointer comparisons
    if (jl_is_leaf_type(type)) {
        if (x.TIndex) {
            unsigned tindex = get_box_tindex((jl_datatype_t*)type, x.typ);
            if (tindex > 0) {
                // optimize more when we know that this is a split union-type where tindex = 0 is invalid
                Value *xtindex = ctx.builder.CreateAnd(x.TIndex, ConstantInt::get(T_int8, 0x7f));
                return std::make_pair(ctx.builder.CreateICmpEQ(xtindex, ConstantInt::get(T_int8, tindex)), false);
            }
            else {
                // test for (x.TIndex == 0x80 && typeof(x.V) == type)
                Value *isboxed = ctx.builder.CreateICmpEQ(x.TIndex, ConstantInt::get(T_int8, 0x80));
                BasicBlock *currBB = ctx.builder.GetInsertBlock();
                BasicBlock *isaBB = BasicBlock::Create(jl_LLVMContext, "isa", ctx.f);
                BasicBlock *postBB = BasicBlock::Create(jl_LLVMContext, "post_isa", ctx.f);
                ctx.builder.CreateCondBr(isboxed, isaBB, postBB);
                ctx.builder.SetInsertPoint(isaBB);
                Value *istype_boxed = ctx.builder.CreateICmpEQ(emit_typeof(ctx, x.V),
                    maybe_decay_untracked(literal_pointer_val(ctx, type)));
                ctx.builder.CreateBr(postBB);
                ctx.builder.SetInsertPoint(postBB);
                PHINode *istype = ctx.builder.CreatePHI(T_int1, 2);
                istype->addIncoming(ConstantInt::get(T_int1, 0), currBB);
                istype->addIncoming(istype_boxed, isaBB);
                return std::make_pair(istype, false);
            }
        }
        return std::make_pair(ctx.builder.CreateICmpEQ(emit_typeof_boxed(ctx, x),
            maybe_decay_untracked(literal_pointer_val(ctx, type))), false);
    }
    // everything else can be handled via subtype tests
    Value *vxt = maybe_decay_untracked(emit_typeof_boxed(ctx, x));
    return std::make_pair(ctx.builder.CreateICmpNE(
            ctx.builder.CreateCall(prepare_call(jlsubtype_func),
              { vxt,
                literal_pointer_val(ctx, type) }),
            ConstantInt::get(T_int32, 0)), false);
}

static void emit_typecheck(jl_codectx_t &ctx, const jl_cgval_t &x, jl_value_t *type, const std::string &msg)
{
    Value *istype;
    bool handled_msg;
    std::tie(istype, handled_msg) = emit_isa(ctx, x, type, &msg);
    if (!handled_msg) {
        BasicBlock *failBB = BasicBlock::Create(jl_LLVMContext, "fail", ctx.f);
        BasicBlock *passBB = BasicBlock::Create(jl_LLVMContext, "pass");
        ctx.builder.CreateCondBr(istype, passBB, failBB);
        ctx.builder.SetInsertPoint(failBB);

        emit_type_error(ctx, x, literal_pointer_val(ctx, type), msg);
        ctx.builder.CreateUnreachable();

        ctx.f->getBasicBlockList().push_back(passBB);
        ctx.builder.SetInsertPoint(passBB);
    }
}

static void emit_leafcheck(jl_codectx_t &ctx, Value *typ, const std::string &msg)
{
    assert(typ->getType() == T_pjlvalue);
    emit_typecheck(ctx, mark_julia_type(ctx, typ, true, jl_any_type, false), (jl_value_t*)jl_datatype_type, msg);
    Value *isleaf;
    isleaf = ctx.builder.CreateConstInBoundsGEP1_32(T_int8, emit_bitcast(ctx, decay_derived(typ), T_pint8), offsetof(jl_datatype_t, isleaftype));
    isleaf = ctx.builder.CreateLoad(isleaf, tbaa_const);
    isleaf = ctx.builder.CreateTrunc(isleaf, T_int1);
    error_unless(ctx, isleaf, msg);
}

#define CHECK_BOUNDS 1
static bool bounds_check_enabled(jl_codectx_t &ctx) {
#if CHECK_BOUNDS==1
    return (!ctx.is_inbounds &&
         jl_options.check_bounds != JL_OPTIONS_CHECK_BOUNDS_OFF) ||
         jl_options.check_bounds == JL_OPTIONS_CHECK_BOUNDS_ON;
#else
    return 0;
#endif
}

static Value *emit_bounds_check(jl_codectx_t &ctx, const jl_cgval_t &ainfo, jl_value_t *ty, Value *i, Value *len)
{
    Value *im1 = ctx.builder.CreateSub(i, ConstantInt::get(T_size, 1));
#if CHECK_BOUNDS==1
    if (bounds_check_enabled(ctx)) {
        Value *ok = ctx.builder.CreateICmpULT(im1, len);
        BasicBlock *failBB = BasicBlock::Create(jl_LLVMContext,"fail",ctx.f);
        BasicBlock *passBB = BasicBlock::Create(jl_LLVMContext,"pass");
        ctx.builder.CreateCondBr(ok, passBB, failBB);
        ctx.builder.SetInsertPoint(failBB);
        if (!ty) { // jl_value_t** tuple (e.g. the vararg)
            ctx.builder.CreateCall(prepare_call(jlvboundserror_func), { ainfo.V, len, i });
        }
        else if (ainfo.isboxed) { // jl_datatype_t or boxed jl_value_t
            ctx.builder.CreateCall(prepare_call(jlboundserror_func), { mark_callee_rooted(boxed(ctx, ainfo)), i });
        }
        else { // unboxed jl_value_t*
            Value *a = ainfo.V;
            if (ainfo.isghost) {
                a = Constant::getNullValue(T_pint8);
            }
            else if (!ainfo.ispointer()) {
                // CreateAlloca is OK here since we are on an error branch
                Value *tempSpace = ctx.builder.CreateAlloca(a->getType());
                ctx.builder.CreateStore(a, tempSpace);
                a = tempSpace;
            }
            ctx.builder.CreateCall(prepare_call(jluboundserror_func), {
                                emit_bitcast(ctx, decay_derived(a), T_pint8),
                                literal_pointer_val(ctx, ty),
                                i });
        }
        ctx.builder.CreateUnreachable();
        ctx.f->getBasicBlockList().push_back(passBB);
        ctx.builder.SetInsertPoint(passBB);
    }
#endif
    return im1;
}

// --- loading and storing ---

static Value *compute_box_tindex(Value *datatype, jl_value_t *supertype, jl_value_t *ut, jl_codectx_t *ctx)
{
    Value *tindex = ConstantInt::get(T_int8, 0);
    unsigned counter = 0;
    for_each_uniontype_small(
            [&](unsigned idx, jl_datatype_t *jt) {
                if (jl_subtype((jl_value_t*)jt, supertype)) {
                    Value *cmp = builder.CreateICmpEQ(literal_pointer_val((jl_value_t*)jt), datatype);
                    tindex = builder.CreateSelect(cmp, ConstantInt::get(T_int8, idx), tindex);
                }
            },
            ut,
            counter);
    return tindex;
}

// get the runtime tindex value
static Value *compute_tindex_unboxed(const jl_cgval_t &val, jl_value_t *typ, jl_codectx_t *ctx)
{
    if (val.constant)
        return ConstantInt::get(T_int8, get_box_tindex((jl_datatype_t*)jl_typeof(val.constant), typ));
    if (val.isboxed)
        return compute_box_tindex(emit_typeof_boxed(val, ctx), val.typ, typ, ctx);
    assert(val.TIndex);
    return builder.CreateAnd(val.TIndex, ConstantInt::get(T_int8, 0x7f));
}

// If given alignment is 0 and LLVM's assumed alignment for a load/store via ptr
// might be stricter than the Julia alignment for jltype, return the alignment of jltype.
// Otherwise return the given alignment.
//
// Parameter ptr should be the pointer argument for the LoadInst or StoreInst.
// It is currently unused, but might be used in the future for a more precise answer.
static unsigned julia_alignment(Value* /*ptr*/, jl_value_t *jltype, unsigned alignment)
{
    if (!alignment) {
        alignment = jl_datatype_align(jltype);
        assert(alignment <= JL_HEAP_ALIGNMENT);
        assert(JL_HEAP_ALIGNMENT % alignment == 0);
    }
    return alignment;
}

static Value *emit_unbox(jl_codectx_t &ctx, Type *to, const jl_cgval_t &x, jl_value_t *jt, Value* dest = NULL, bool volatile_store = false);

static jl_cgval_t typed_load(jl_codectx_t &ctx, Value *ptr, Value *idx_0based, jl_value_t *jltype,
                             MDNode *tbaa, bool maybe_null_if_boxed = true, unsigned alignment = 0)
{
    bool isboxed;
    Type *elty = julia_type_to_llvm(jltype, &isboxed);
    if (type_is_ghost(elty))
        return ghostValue(jltype);
    Value *data;
    if (isboxed)
        elty = T_prjlvalue;
    // TODO: preserving_pointercast?
    if (ptr->getType()->getContainedType(0) != elty)
        data = emit_bitcast(ctx, ptr, PointerType::get(elty, 0));
    else
        data = ptr;
    if (idx_0based)
        data = ctx.builder.CreateGEP(data, idx_0based);
    Value *elt;
    // TODO: can only lazy load if we can create a gc root for ptr for the lifetime of elt
    //if (elty->isAggregateType() && tbaa == tbaa_immut && !alignment) { // can lazy load on demand, no copy needed
    //    elt = data;
    //}
    //else {
        Instruction *load = ctx.builder.CreateAlignedLoad(data, isboxed ?
            alignment : julia_alignment(data, jltype, alignment), false);
        if (isboxed)
            load = maybe_mark_load_dereferenceable(load, true, jltype);
        if (tbaa) {
            elt = tbaa_decorate(tbaa, load);
        }
        else {
            elt = load;
        }
        if (maybe_null_if_boxed && isboxed) {
            null_pointer_check(ctx, elt);
        }
    //}
    return mark_julia_type(ctx, elt, isboxed, jltype);
}

static void typed_store(jl_codectx_t &ctx,
        Value *ptr, Value *idx_0based, const jl_cgval_t &rhs,
        jl_value_t *jltype, MDNode *tbaa,
        Value *parent,  // for the write barrier, NULL if no barrier needed
        unsigned alignment = 0, bool root_box = true) // if the value to store needs a box, should we root it ?
{
    bool isboxed;
    Type *elty = julia_type_to_llvm(jltype, &isboxed);
    if (type_is_ghost(elty))
        return;
    Value *r;
    if (!isboxed) {
        r = emit_unbox(ctx, elty, rhs, jltype);
    }
    else {
        r = maybe_decay_untracked(boxed(ctx, rhs, root_box));
        if (parent != NULL)
            emit_write_barrier(ctx, parent, r);
    }
    Value *data;
    if (ptr->getType()->getContainedType(0) != elty) {
        if (isboxed) {
            data = emit_bitcast(ctx, ptr, T_pprjlvalue);
        } else {
            data = emit_bitcast(ctx, ptr, PointerType::get(elty, cast<PointerType>(ptr->getType())->getAddressSpace()));
        }
    } else
        data = ptr;
    Instruction *store = ctx.builder.CreateAlignedStore(r, ctx.builder.CreateGEP(data,
        idx_0based), isboxed ? alignment : julia_alignment(r, jltype, alignment));
    if (tbaa)
        tbaa_decorate(tbaa, store);
}

// --- convert boolean value to julia ---

static Value *julia_bool(jl_codectx_t &ctx, Value *cond)
{
    return ctx.builder.CreateSelect(cond, literal_pointer_val(ctx, jl_true),
                                      literal_pointer_val(ctx, jl_false));
}

// --- get the inferred type of an AST node ---

static inline jl_module_t *topmod(jl_codectx_t &ctx)
{
    return jl_base_relative_to(ctx.module);
}

static jl_value_t *expr_type(jl_codectx_t &ctx, jl_value_t *e)
{
    if (jl_is_ssavalue(e)) {
        if (jl_is_long(ctx.source->ssavaluetypes))
            return (jl_value_t*)jl_any_type;
        int idx = ((jl_ssavalue_t*)e)->id;
        assert(jl_is_array(ctx.source->ssavaluetypes));
        jl_array_t *ssavalue_types = (jl_array_t*)ctx.source->ssavaluetypes;
        return jl_array_ptr_ref(ssavalue_types, idx);
    }
    if (jl_typeis(e, jl_slotnumber_type)) {
        jl_array_t *slot_types = (jl_array_t*)ctx.source->slottypes;
        if (!jl_is_array(slot_types))
            return (jl_value_t*)jl_any_type;
        return jl_array_ptr_ref(slot_types, jl_slot_number(e)-1);
    }
    if (jl_typeis(e, jl_typedslot_type)) {
        jl_value_t *typ = jl_typedslot_get_type(e);
        if (jl_is_typevar(typ))
            typ = ((jl_tvar_t*)typ)->ub;
        return typ;
    }
    if (jl_is_expr(e)) {
        if (((jl_expr_t*)e)->head == static_parameter_sym) {
            size_t idx = jl_unbox_long(jl_exprarg(e,0))-1;
            if (idx >= jl_svec_len(ctx.linfo->sparam_vals))
                return (jl_value_t*)jl_any_type;
            e = jl_svecref(ctx.linfo->sparam_vals, idx);
            if (jl_is_typevar(e))
                return (jl_value_t*)jl_any_type;
            goto type_of_constant;
        }
        jl_value_t *typ = ((jl_expr_t*)e)->etype;
        if (jl_is_typevar(typ))
            typ = ((jl_tvar_t*)typ)->ub;
        return typ;
    }
    if (jl_is_quotenode(e)) {
        e = jl_fieldref(e,0);
        goto type_of_constant;
    }
    if (jl_is_globalref(e)) {
        jl_sym_t *s = (jl_sym_t*)jl_globalref_name(e);
        jl_binding_t *b = jl_get_binding(jl_globalref_mod(e), s);
        if (b && b->constp) {
            e = b->value;
            goto type_of_constant;
        }
        return (jl_value_t*)jl_any_type;
    }
    if (jl_is_symbol(e)) {
        jl_binding_t *b = jl_get_binding(ctx.module, (jl_sym_t*)e);
        if (!b || !b->value)
            return (jl_value_t*)jl_any_type;
        if (b->constp)
            e = b->value;
        else
            return (jl_value_t*)jl_any_type;
    }
type_of_constant:
    if (jl_is_type(e))
        return (jl_value_t*)jl_wrap_Type(e);
    return (jl_value_t*)jl_typeof(e);
}

// --- accessing the representations of built-in data types ---

static Constant *julia_const_to_llvm(jl_value_t *e);
static Value *data_pointer(jl_codectx_t &ctx, const jl_cgval_t &x, Type *astype = T_ppjlvalue)
{
    Value *data = x.V;
    if (x.constant) {
        Constant *val = julia_const_to_llvm(x.constant);
        if (val) {
            data = get_pointer_to_constant(val, "", *jl_Module);
        }
        else {
            data = boxed(ctx, x);
        }
    }
    if (data->getType() != astype)
        data = emit_bitcast(ctx, data, astype);
    return decay_derived(data);
}

static bool emit_getfield_unknownidx(jl_codectx_t &ctx,
        jl_cgval_t *ret, const jl_cgval_t &strct,
        Value *idx, jl_datatype_t *stt)
{
    size_t nfields = jl_datatype_nfields(stt);
    if (strct.ispointer()) { // boxed or stack
        if (is_datatype_all_pointers(stt)) {
            idx = emit_bounds_check(ctx, strct, (jl_value_t*)stt, idx, ConstantInt::get(T_size, nfields));
            bool maybe_null = (unsigned)stt->ninitialized != nfields;
            size_t minimum_field_size = (size_t)-1;
            for (size_t i = 0; i < nfields; ++i) {
                minimum_field_size = std::min(minimum_field_size,
                    dereferenceable_size(jl_field_type(stt, i)));
                if (minimum_field_size == 0)
                    break;
            }
            Value *fld = tbaa_decorate(strct.tbaa,
                maybe_mark_load_dereferenceable(
                    ctx.builder.CreateLoad(
                        ctx.builder.CreateBitCast(ctx.builder.CreateGEP(decay_derived(data_pointer(ctx, strct)), idx),
                            PointerType::get(T_prjlvalue, AddressSpace::Derived))),
                    maybe_null,  minimum_field_size));
            if (maybe_null)
                null_pointer_check(ctx, fld);
            *ret = mark_julia_type(ctx, fld, true, jl_any_type, strct.gcroot || !strct.isimmutable);
            return true;
        }
        else if (is_tupletype_homogeneous(stt->types)) {
            assert(nfields > 0); // nf == 0 trapped by all_pointers case
            jl_value_t *jt = jl_field_type(stt, 0);
            idx = emit_bounds_check(ctx, strct, (jl_value_t*)stt, idx, ConstantInt::get(T_size, nfields));
            Value *ptr = data_pointer(ctx, strct);
            if (!stt->mutabl) {
                // just compute the pointer and let user load it when necessary
                Type *fty = julia_type_to_llvm(jt);
                Value *addr = ctx.builder.CreateGEP(emit_bitcast(ctx, decay_derived(ptr), PointerType::get(fty,0)), idx);
                *ret = mark_julia_slot(addr, jt, NULL, strct.tbaa);
                ret->gcroot = strct.gcroot;
                ret->isimmutable = strct.isimmutable;
                return true;
            }
            *ret = typed_load(ctx, ptr, idx, jt, strct.tbaa, false);
            return true;
        }
        else if (strct.isboxed) {
            idx = ctx.builder.CreateSub(idx, ConstantInt::get(T_size, 1));
            Value *fld = ctx.builder.CreateCall(prepare_call(jlgetnthfieldchecked_func), { boxed(ctx, strct), idx });
            *ret = mark_julia_type(ctx, fld, true, jl_any_type);
            return true;
        }
    }
    else if (is_tupletype_homogeneous(stt->types)) {
        assert(jl_isbits(stt));
        if (nfields == 0) {
            idx = emit_bounds_check(ctx, ghostValue(stt),
                                    (jl_value_t*)stt, idx, ConstantInt::get(T_size, nfields));
            *ret = jl_cgval_t();
            return true;
        }
        assert(!jl_field_isptr(stt, 0));
        jl_value_t *jt = jl_field_type(stt, 0);
        Value *idx0 = emit_bounds_check(ctx, strct, (jl_value_t*)stt, idx, ConstantInt::get(T_size, nfields));
        if (strct.isghost) {
            *ret = ghostValue(jt);
            return true;
        }
        // llvm::VectorType
        if (sizeof(void*) != sizeof(int))
            idx0 = ctx.builder.CreateTrunc(idx0, T_int32); // llvm3.3 requires this, harmless elsewhere
        Value *fld = ctx.builder.CreateExtractElement(strct.V, idx0);
        *ret = mark_julia_type(ctx, fld, false, jt);
        return true;
    }
    return false;
}

static jl_cgval_t emit_getfield_knownidx(jl_codectx_t &ctx, const jl_cgval_t &strct, unsigned idx, jl_datatype_t *jt)
{
    jl_value_t *jfty = jl_field_type(jt, idx);
    Type *elty = julia_type_to_llvm(jfty);
    if (jfty == jl_bottom_type) {
        raise_exception(ctx, literal_pointer_val(ctx, jl_undefref_exception));
        return jl_cgval_t(); // unreachable
    }
    if (type_is_ghost(elty))
        return ghostValue(jfty);
    Value *fldv = NULL;
    if (strct.ispointer()) {
        Value *addr;
        bool isboxed;
        Type *lt = julia_type_to_llvm((jl_value_t*)jt, &isboxed);
        if (isboxed) {
            Value *ptr = decay_derived(data_pointer(ctx, strct, T_pint8));
            Value *llvm_idx = ConstantInt::get(T_size, jl_field_offset(jt, idx));
            addr = ctx.builder.CreateGEP(ptr, llvm_idx);
        }
        else {
            if (VectorType *vlt = dyn_cast<VectorType>(lt)) {
                // doesn't have the struct wrapper, so this must have been a VecElement
                // cast to the element type so that it can be addressed with GEP
                lt = vlt->getElementType();
                Value *ptr = data_pointer(ctx, strct, lt->getPointerTo());
                Value *llvm_idx = ConstantInt::get(T_size, idx);
                addr = ctx.builder.CreateGEP(lt, ptr, llvm_idx);
            }
            else if (lt->isSingleValueType()) {
                addr = data_pointer(ctx, strct, lt->getPointerTo());
            }
            else {
                Value *ptr = data_pointer(ctx, strct, lt->getPointerTo());
                addr = ctx.builder.CreateStructGEP(lt, ptr, idx);
            }
        }
        int align = jl_field_offset(jt, idx);
        align |= 16;
        align &= -align;
        if (jl_field_isptr(jt, idx)) {
            bool maybe_null = idx >= (unsigned)jt->ninitialized;
            Instruction *Load = maybe_mark_load_dereferenceable(
                ctx.builder.CreateLoad(emit_bitcast(ctx, addr, T_pprjlvalue)),
                maybe_null, jl_field_type(jt, idx)
            );
            Value *fldv = tbaa_decorate(strct.tbaa, Load);
            if (maybe_null)
                null_pointer_check(ctx, fldv);
            return mark_julia_type(ctx, fldv, true, jfty, strct.gcroot || !strct.isimmutable);
        }
        else if (jl_is_uniontype(jfty)) {
            int fsz = jl_field_size(jt, idx);
            Value *ptindex = builder.CreateGEP(LLVM37_param(T_int8) emit_bitcast(addr, T_pint8), ConstantInt::get(T_size, fsz - 1));
            Value *tindex = builder.CreateNUWAdd(ConstantInt::get(T_int8, 1), builder.CreateLoad(ptindex));
            bool isimmutable = strct.isimmutable;
            Value *gcroot = strct.gcroot;
            if (jt->mutabl) {
                // move value to an immutable stack slot
                Type *AT = ArrayType::get(IntegerType::get(jl_LLVMContext, 8 * align), (fsz + align - 2) / align);
                AllocaInst *lv = emit_static_alloca(AT, ctx);
                if (align > 1)
                    lv->setAlignment(align);
                Value *nbytes = ConstantInt::get(T_size, fsz - 1);
                builder.CreateMemCpy(lv, addr, nbytes, align);
                addr = lv;
                isimmutable = true;
                gcroot = NULL;
            }
            jl_cgval_t fieldval = mark_julia_slot(addr, jfty, tindex, strct.tbaa);
            fieldval.isimmutable = isimmutable;
            fieldval.gcroot = gcroot;
            return fieldval;
        }
        else if (!jt->mutabl) {
            // just compute the pointer and let user load it when necessary
            jl_cgval_t fieldval = mark_julia_slot(addr, jfty, NULL, strct.tbaa);
            fieldval.isimmutable = strct.isimmutable;
            fieldval.gcroot = strct.gcroot;
            return fieldval;
        }
        int align = jl_field_offset(jt, idx);
        align |= 16;
        align &= -align;
        return typed_load(ctx, addr, ConstantInt::get(T_size, 0), jfty, strct.tbaa, true, align);
    }
    else if (isa<UndefValue>(strct.V)) {
        return jl_cgval_t();
    }
    else {
        if (strct.V->getType()->isVectorTy()) {
            fldv = ctx.builder.CreateExtractElement(strct.V, ConstantInt::get(T_int32, idx));
        }
        else {
            // VecElement types are unwrapped in LLVM.
            assert( strct.V->getType()->isSingleValueType() );
            fldv = strct.V;
        }
        assert(!jl_field_isptr(jt, idx));
        return mark_julia_type(ctx, fldv, false, jfty);
    }
}

// emit length of vararg tuple
static Value *emit_n_varargs(jl_codectx_t &ctx)
{
    int nreq = ctx.nReqArgs;
    Value *valen = ctx.builder.CreateSub((Value*)ctx.argCount,
                                     ConstantInt::get(T_int32, nreq));
#ifdef _P64
    return ctx.builder.CreateSExt(valen, T_int64);
#else
    return valen;
#endif
}

static bool arraytype_constshape(jl_value_t *ty)
{
    return (jl_is_array_type(ty) && jl_is_leaf_type(ty) &&
            jl_is_long(jl_tparam1(ty)) && jl_unbox_long(jl_tparam1(ty)) != 1);
}

static void maybe_alloc_arrayvar(jl_codectx_t &ctx, int s)
{
    jl_value_t *jt = ctx.slots[s].value.typ;
    if (arraytype_constshape(jt)) {
        // TODO: this optimization does not yet work with 1-d arrays, since the
        // length and data pointer can change at any time via push!
        // we could make it work by reloading the metadata when the array is
        // passed to an external function (ideally only impure functions)
        int ndims = jl_unbox_long(jl_tparam1(jt));
        jl_value_t *jelt = jl_tparam0(jt);
        bool isboxed = !jl_array_store_unboxed(jelt);
        Type *elt = julia_type_to_llvm(jelt);
        if (type_is_ghost(elt))
            return;
        if (isboxed)
            elt = T_prjlvalue;
        // CreateAlloca is OK here because maybe_alloc_arrayvar is only called in the prologue setup
        jl_arrayvar_t &av = (*ctx.arrayvars)[s];
        av.dataptr = ctx.builder.CreateAlloca(PointerType::get(elt, 0));
        av.len = ctx.builder.CreateAlloca(T_size);
        for (int i = 0; i < ndims - 1; i++)
            av.sizes.push_back(ctx.builder.CreateAlloca(T_size));
        av.ty = jt;
    }
}

static Value *emit_arraysize(jl_codectx_t &ctx, const jl_cgval_t &tinfo, Value *dim)
{
    Value *t = boxed(ctx, tinfo);
    int o = offsetof(jl_array_t, nrows) / sizeof(void*) - 1;
    MDNode *tbaa = arraytype_constshape(tinfo.typ) ? tbaa_const : tbaa_arraysize;
    return emit_nthptr_recast(ctx,
            t,
            ctx.builder.CreateAdd(dim, ConstantInt::get(dim->getType(), o)),
            tbaa, T_psize);
}

static jl_arrayvar_t *arrayvar_for(jl_codectx_t &ctx, jl_value_t *ex)
{
    if (ex == NULL)
        return NULL;
    if (!jl_is_slot(ex))
        return NULL;
    int sl = jl_slot_number(ex) - 1;
    auto av = ctx.arrayvars->find(sl);
    if (av != ctx.arrayvars->end())
        return &av->second;
    //TODO: ssavalue case
    return NULL;
}

static Value *emit_arraysize(jl_codectx_t &ctx, const jl_cgval_t &tinfo, int dim)
{
    return emit_arraysize(ctx, tinfo, ConstantInt::get(T_int32, dim));
}

static Value *emit_arraylen_prim(jl_codectx_t &ctx, const jl_cgval_t &tinfo)
{
    Value *t = boxed(ctx, tinfo);
    jl_value_t *ty = tinfo.typ;
#ifdef STORE_ARRAY_LEN
    Value *addr = ctx.builder.CreateStructGEP(jl_array_llvmt,
                                          emit_bitcast(ctx, decay_derived(t), jl_parray_llvmt),
                                          1); //index (not offset) of length field in jl_parray_llvmt

    MDNode *tbaa = arraytype_constshape(ty) ? tbaa_const : tbaa_arraylen;
    return tbaa_decorate(tbaa, ctx.builder.CreateLoad(addr, false));
#else
    jl_value_t *p1 = jl_tparam1(ty); // FIXME: check that ty is an array type
    if (jl_is_long(p1)) {
        size_t nd = jl_unbox_long(p1);
        Value *l = ConstantInt::get(T_size, 1);
        for(size_t i=0; i < nd; i++) {
            l = ctx.builder.CreateMul(l, emit_arraysize(ctx, t, (int)(i + 1)));
        }
        return l;
    }
    else {
        std::vector<Type *> fargt(0);
        fargt.push_back(T_pjlvalue);
        FunctionType *ft = FunctionType::get(T_size, fargt, false);
        Value *alen = jl_Module->getOrInsertFunction("jl_array_len_", ft); // TODO: move to codegen init block
        return ctx.builder.CreateCall(prepare_call(alen), t);
    }
#endif
}

static Value *emit_arraylen(jl_codectx_t &ctx, const jl_cgval_t &tinfo, jl_value_t *ex)
{
    jl_arrayvar_t *av = arrayvar_for(ctx, ex);
    if (av != NULL)
        return ctx.builder.CreateLoad(av->len);
    return emit_arraylen_prim(ctx, tinfo);
}

static Value *emit_arrayptr(jl_codectx_t &ctx, const jl_cgval_t &tinfo, bool isboxed = false)
{
    Value *t = boxed(ctx, tinfo);
    Value *addr = ctx.builder.CreateStructGEP(jl_array_llvmt,
                                          emit_bitcast(ctx, decay_derived(t), jl_parray_llvmt),
                                          0); //index (not offset) of data field in jl_parray_llvmt

    MDNode *tbaa = arraytype_constshape(tinfo.typ) ? tbaa_const : tbaa_arrayptr;
    if (isboxed) {
        addr = ctx.builder.CreateBitCast(addr,
            PointerType::get(T_pprjlvalue, cast<PointerType>(addr->getType())->getAddressSpace()));
    }
    return tbaa_decorate(tbaa, ctx.builder.CreateLoad(addr, false));
}

static Value *emit_arrayptr(jl_codectx_t &ctx, const jl_cgval_t &tinfo, jl_value_t *ex, bool isboxed = false)
{
    jl_arrayvar_t *av = arrayvar_for(ctx, ex);
    if (av!=NULL)
        return ctx.builder.CreateLoad(av->dataptr);
    return emit_arrayptr(ctx, tinfo, isboxed);
}

static Value *emit_arraysize(jl_codectx_t &ctx, const jl_cgval_t &tinfo, jl_value_t *ex, int dim)
{
    jl_arrayvar_t *av = arrayvar_for(ctx, ex);
    if (av != NULL && dim <= (int)av->sizes.size())
        return ctx.builder.CreateLoad(av->sizes[dim - 1]);
    return emit_arraysize(ctx, tinfo, dim);
}

static Value *emit_arrayflags(jl_codectx_t &ctx, const jl_cgval_t &tinfo)
{
    Value *t = boxed(ctx, tinfo);
#ifdef STORE_ARRAY_LEN
    int arrayflag_field = 2;
#else
    int arrayflag_field = 1;
#endif
    Value *addr = ctx.builder.CreateStructGEP(
            jl_array_llvmt,
            emit_bitcast(ctx, decay_derived(t), jl_parray_llvmt),
            arrayflag_field);
    return tbaa_decorate(tbaa_arrayflags, ctx.builder.CreateLoad(addr));
}

static Value *emit_arrayelsize(jl_codectx_t &ctx, const jl_cgval_t &tinfo)
{
    Value *t = boxed(ctx, tinfo);
#ifdef STORE_ARRAY_LEN
    int elsize_field = 3;
#else
    int elsize_field = 2;
#endif
    Value *addr = ctx.builder.CreateStructGEP(jl_array_llvmt,
                                          emit_bitcast(ctx, decay_derived(t), jl_parray_llvmt),
                                          elsize_field);
    return tbaa_decorate(tbaa_const, ctx.builder.CreateLoad(addr));
}

static void assign_arrayvar(jl_codectx_t &ctx, jl_arrayvar_t &av, const jl_cgval_t &ainfo)
{
    Value *aptr = emit_bitcast(ctx,
        emit_arrayptr(ctx, ainfo),
        av.dataptr->getType()->getContainedType(0));
    tbaa_decorate(tbaa_arrayptr, ctx.builder.CreateStore(aptr, av.dataptr));
    ctx.builder.CreateStore(emit_arraylen_prim(ctx, ainfo), av.len);
    for (size_t i = 0; i < av.sizes.size(); i++)
        ctx.builder.CreateStore(emit_arraysize(ctx, ainfo, i + 1), av.sizes[i]);
}

// Returns the size of the array represented by `tinfo` for the given dimension `dim` if
// `dim` is a valid dimension, otherwise returns constant one.
static Value *emit_arraysize_for_unsafe_dim(jl_codectx_t &ctx,
        const jl_cgval_t &tinfo, jl_value_t *ex, size_t dim, size_t nd)
{
    return dim > nd ? ConstantInt::get(T_size, 1) : emit_arraysize(ctx, tinfo, ex, dim);
}

// `nd == -1` means the dimension is unknown.
static Value *emit_array_nd_index(jl_codectx_t &ctx,
        const jl_cgval_t &ainfo, jl_value_t *ex, ssize_t nd, jl_value_t **args, size_t nidxs)
{
    Value *a = boxed(ctx, ainfo);
    Value *i = ConstantInt::get(T_size, 0);
    Value *stride = ConstantInt::get(T_size, 1);
#if CHECK_BOUNDS==1
    bool bc = (!ctx.is_inbounds &&
               jl_options.check_bounds != JL_OPTIONS_CHECK_BOUNDS_OFF) ||
        jl_options.check_bounds == JL_OPTIONS_CHECK_BOUNDS_ON;
    BasicBlock *failBB=NULL, *endBB=NULL;
    if (bc) {
        failBB = BasicBlock::Create(jl_LLVMContext, "oob");
        endBB = BasicBlock::Create(jl_LLVMContext, "idxend");
    }
#endif
    Value **idxs = (Value**)alloca(sizeof(Value*)*nidxs);
    for (size_t k = 0; k < nidxs; k++) {
        idxs[k] = emit_unbox(ctx, T_size, emit_expr(ctx, args[k]), NULL);
    }
    Value *ii;
    for (size_t k = 0; k < nidxs; k++) {
        ii = ctx.builder.CreateSub(idxs[k], ConstantInt::get(T_size, 1));
        i = ctx.builder.CreateAdd(i, ctx.builder.CreateMul(ii, stride));
        if (k < nidxs - 1) {
            assert(nd >= 0);
            Value *d = emit_arraysize_for_unsafe_dim(ctx, ainfo, ex, k + 1, nd);
#if CHECK_BOUNDS==1
            if (bc) {
                BasicBlock *okBB = BasicBlock::Create(jl_LLVMContext, "ib");
                // if !(i < d) goto error
                ctx.builder.CreateCondBr(ctx.builder.CreateICmpULT(ii, d), okBB, failBB);
                ctx.f->getBasicBlockList().push_back(okBB);
                ctx.builder.SetInsertPoint(okBB);
            }
#endif
            stride = ctx.builder.CreateMul(stride, d);
        }
    }
#if CHECK_BOUNDS==1
    if (bc) {
        // We have already emitted a bounds check for each index except for
        // the last one which we therefore have to do here.
        bool linear_indexing = nd == -1 || nidxs < (size_t)nd;
        if (linear_indexing) {
            // Compare the linearized index `i` against the linearized size of
            // the accessed array, i.e. `if !(i < alen) goto error`.
            if (nidxs > 1) {
                // TODO: REMOVE DEPWARN AND RETURN FALSE AFTER 0.6.
                // We need to check if this is inside the non-linearized size
                BasicBlock *partidx = BasicBlock::Create(jl_LLVMContext, "partlinidx");
                BasicBlock *partidxwarn = BasicBlock::Create(jl_LLVMContext, "partlinidxwarn");
                Value *d = emit_arraysize_for_unsafe_dim(ctx, ainfo, ex, nidxs, nd);
                ctx.builder.CreateCondBr(ctx.builder.CreateICmpULT(ii, d), endBB, partidx);

                // We failed the normal bounds check; check to see if we're
                // inside the linearized size (partial linear indexing):
                ctx.f->getBasicBlockList().push_back(partidx);
                ctx.builder.SetInsertPoint(partidx);
                Value *alen = emit_arraylen(ctx, ainfo, ex);
                ctx.builder.CreateCondBr(ctx.builder.CreateICmpULT(i, alen), partidxwarn, failBB);

                // We passed the linearized bounds check; now throw the depwarn:
                ctx.f->getBasicBlockList().push_back(partidxwarn);
                ctx.builder.SetInsertPoint(partidxwarn);
                ctx.builder.CreateCall(prepare_call(jldepwarnpi_func), ConstantInt::get(T_size, nidxs));
                ctx.builder.CreateBr(endBB);
            } else {
                Value *alen = emit_arraylen(ctx, ainfo, ex);
                ctx.builder.CreateCondBr(ctx.builder.CreateICmpULT(i, alen), endBB, failBB);
            }
        } else {
            // Compare the last index of the access against the last dimension of
            // the accessed array, i.e. `if !(last_index < last_dimension) goto error`.
            assert(nd >= 0);
            Value *last_index = ii;
            Value *last_dimension = emit_arraysize_for_unsafe_dim(ctx, ainfo, ex, nidxs, nd);
            ctx.builder.CreateCondBr(ctx.builder.CreateICmpULT(last_index, last_dimension), endBB, failBB);
        }

        ctx.f->getBasicBlockList().push_back(failBB);
        ctx.builder.SetInsertPoint(failBB);
        // CreateAlloca is OK here since we are on an error branch
        Value *tmp = ctx.builder.CreateAlloca(T_size, ConstantInt::get(T_size, nidxs));
        for(size_t k=0; k < nidxs; k++) {
            ctx.builder.CreateStore(idxs[k], ctx.builder.CreateGEP(tmp, ConstantInt::get(T_size, k)));
        }
        ctx.builder.CreateCall(prepare_call(jlboundserrorv_func),
            { mark_callee_rooted(a), tmp, ConstantInt::get(T_size, nidxs) });
        ctx.builder.CreateUnreachable();

        ctx.f->getBasicBlockList().push_back(endBB);
        ctx.builder.SetInsertPoint(endBB);
    }
#endif

    return i;
}

// --- boxing ---

static Value *emit_allocobj(jl_codectx_t &ctx, size_t static_size, Value *jt);

static void init_bits_value(jl_codectx_t &ctx, Value *newv, Value *v, MDNode *tbaa,
                            unsigned alignment = sizeof(void*)) // min alignment in julia's gc is pointer-aligned
{
    // newv should already be tagged
    tbaa_decorate(tbaa, ctx.builder.CreateAlignedStore(v, emit_bitcast(ctx, newv,
        PointerType::get(v->getType(), 0)), alignment));
}

static void init_bits_cgval(jl_codectx_t &ctx, Value *newv, const jl_cgval_t& v, MDNode *tbaa)
{
    // newv should already be tagged
    if (v.ispointer()) {
        ctx.builder.CreateMemCpy(newv, data_pointer(ctx, v, T_pint8), jl_datatype_size(v.typ), sizeof(void*));
    }
    else {
        init_bits_value(ctx, newv, v.V, tbaa);
    }
}

static jl_value_t *static_constant_instance(Constant *constant, jl_value_t *jt)
{
    assert(constant != NULL);

    if (isa<UndefValue>(constant))
        return NULL;

    ConstantInt *cint = dyn_cast<ConstantInt>(constant);
    if (cint != NULL) {
        assert(jl_is_datatype(jt));
        if (jt == (jl_value_t*)jl_bool_type)
            return cint->isZero() ? jl_false : jl_true;
        return jl_new_bits(jt,
            const_cast<uint64_t *>(cint->getValue().getRawData()));
    }

    ConstantFP *cfp = dyn_cast<ConstantFP>(constant);
    if (cfp != NULL) {
        assert(jl_is_datatype(jt));
        return jl_new_bits(jt,
            const_cast<uint64_t *>(cfp->getValueAPF().bitcastToAPInt().getRawData()));
    }

    ConstantPointerNull *cpn = dyn_cast<ConstantPointerNull>(constant);
    if (cpn != NULL) {
        assert(jl_is_cpointer_type(jt));
        uint64_t val = 0;
        return jl_new_bits(jt,&val);
    }

    // issue #8464
    ConstantExpr *ce = dyn_cast<ConstantExpr>(constant);
    if (ce != NULL) {
        if (ce->isCast()) {
            return static_constant_instance(dyn_cast<Constant>(ce->getOperand(0)), jt);
        }
    }

    size_t nargs = 0;
    if (ConstantStruct *cst = dyn_cast<ConstantStruct>(constant))
        nargs = cst->getType()->getNumElements();
    else if (ConstantVector *cvec = dyn_cast<ConstantVector>(constant))
        nargs = cvec->getType()->getNumElements();
    else if (ConstantArray *carr = dyn_cast<ConstantArray>(constant))
        nargs = carr->getType()->getNumElements();
    else if (ConstantDataVector *cdv = dyn_cast<ConstantDataVector>(constant))
        nargs = cdv->getType()->getNumElements();
    else if (isa<Function>(constant))
        return NULL;
    else
        assert(false && "Cannot process this type of constant");

    assert(jl_is_tuple_type(jt));

    jl_value_t **tupleargs;
    JL_GC_PUSHARGS(tupleargs, nargs);
    for(size_t i=0; i < nargs; i++) {
        tupleargs[i] = static_constant_instance(constant->getAggregateElement(i), jl_tparam(jt,i));
    }
    jl_value_t *tpl = jl_f_tuple(NULL, tupleargs, nargs);
    JL_GC_POP();
    return tpl;
}

static Value *call_with_signed(jl_codectx_t &ctx, Function *sfunc, Value *v)
{
    CallInst *Call = ctx.builder.CreateCall(prepare_call(sfunc), v);
    Call->addAttribute(1, Attribute::SExt);
    return Call;
}

static Value *call_with_unsigned(jl_codectx_t &ctx, Function *ufunc, Value *v)
{
    CallInst *Call = ctx.builder.CreateCall(prepare_call(ufunc), v);
    Call->addAttribute(1, Attribute::ZExt);
    return Call;
}

static void jl_add_method_root(jl_codectx_t &ctx, jl_value_t *val);

static Value *as_value(jl_codectx_t &ctx, Type *to, const jl_cgval_t &v)
{
    assert(!v.isboxed);
    return emit_unbox(ctx, to, v, v.typ);
}

// some types have special boxing functions with small-value caches
static Value *_boxed_special(jl_codectx_t &ctx, const jl_cgval_t &vinfo, Type *t)
{
    jl_value_t *jt = vinfo.typ;
    if (jt == (jl_value_t*)jl_bool_type)
        return julia_bool(ctx, ctx.builder.CreateTrunc(as_value(ctx, t, vinfo), T_int1));
    if (t == T_int1)
        return julia_bool(ctx, as_value(ctx, t, vinfo));

    if (ctx.linfo && jl_is_method(ctx.linfo->def.method) && !vinfo.ispointer()) { // don't bother codegen pre-boxing for toplevel
        if (Constant *c = dyn_cast<Constant>(vinfo.V)) {
            jl_value_t *s = static_constant_instance(c, jt);
            if (s) {
                jl_add_method_root(ctx, s);
                return literal_pointer_val(ctx, s);
            }
        }
    }

    jl_datatype_t *jb = (jl_datatype_t*)jt;
    assert(jl_is_datatype(jb));
    Value *box = NULL;
    if (jb == jl_int8_type)
        box = call_with_signed(ctx, box_int8_func, as_value(ctx, t, vinfo));
    else if (jb == jl_int16_type)
        box = call_with_signed(ctx, box_int16_func, as_value(ctx, t, vinfo));
    else if (jb == jl_int32_type)
        box = call_with_signed(ctx, box_int32_func, as_value(ctx, t, vinfo));
    else if (jb == jl_int64_type)
        box = call_with_signed(ctx, box_int64_func, as_value(ctx, t, vinfo));
    else if (jb == jl_float32_type)
        box = ctx.builder.CreateCall(prepare_call(box_float32_func), as_value(ctx, t, vinfo));
    //if (jb == jl_float64_type)
    //  box = ctx.builder.CreateCall(box_float64_func, as_value(ctx, t, vinfo);
    // for Float64, fall through to generic case below, to inline alloc & init of Float64 box. cheap, I know.
    else if (jb == jl_uint8_type)
        box = call_with_unsigned(ctx, box_uint8_func, as_value(ctx, t, vinfo));
    else if (jb == jl_uint16_type)
        box = call_with_unsigned(ctx, box_uint16_func, as_value(ctx, t, vinfo));
    else if (jb == jl_uint32_type)
        box = call_with_unsigned(ctx, box_uint32_func, as_value(ctx, t, vinfo));
    else if (jb == jl_uint64_type)
        box = call_with_unsigned(ctx, box_uint64_func, as_value(ctx, t, vinfo));
    else if (jb == jl_char_type)
        box = call_with_unsigned(ctx, box_char_func, as_value(ctx, t, vinfo));
    else if (jb == jl_ssavalue_type) {
        unsigned zero = 0;
        Value *v = as_value(ctx, t, vinfo);
        assert(v->getType() == jl_ssavalue_type->struct_decl);
        v = ctx.builder.CreateExtractValue(v, makeArrayRef(&zero, 1));
        box = call_with_unsigned(ctx, box_ssavalue_func, v);
    }
    else if (!jb->abstract && jl_datatype_nbits(jb) == 0) {
        // singleton
        assert(jb->instance != NULL);
        return literal_pointer_val(ctx, jb->instance);
    }
    return box;
}



static Value *box_union(jl_codectx_t &ctx, const jl_cgval_t &vinfo, const SmallBitVector &skip)
{
    // given vinfo::Union{T, S}, emit IR of the form:
    //   ...
    //   switch <tindex>, label <box_union_isboxed> [ 1, label <box_union_1>
    //                                                2, label <box_union_2> ]
    // box_union_1:
    //   box1 = create_box(T)
    //   br post_box_union
    // box_union_2:
    //   box2 = create_box(S)
    //   br post_box_union
    // box_union_isboxed:
    //   br post_box_union
    // post_box_union:
    //   box = phi [ box1, box_union_1 ], [ box2, box_union_2 ], [ vinfo, box_union_isboxed ]
    //   ...
    Value *tindex = vinfo.TIndex;
    BasicBlock *defaultBB = BasicBlock::Create(jl_LLVMContext, "box_union_isboxed", ctx.f);
    SwitchInst *switchInst = ctx.builder.CreateSwitch(tindex, defaultBB);
    BasicBlock *postBB = BasicBlock::Create(jl_LLVMContext, "post_box_union", ctx.f);
    ctx.builder.SetInsertPoint(postBB);
    PHINode *box_merge = ctx.builder.CreatePHI(T_prjlvalue, 2);
    unsigned counter = 0;
    for_each_uniontype_small(
            [&](unsigned idx, jl_datatype_t *jt) {
                if (idx < skip.size() && skip[idx])
                    return;
                Type *t = julia_type_to_llvm((jl_value_t*)jt);
                BasicBlock *tempBB = BasicBlock::Create(jl_LLVMContext, "box_union", ctx.f);
                ctx.builder.SetInsertPoint(tempBB);
                switchInst->addCase(ConstantInt::get(T_int8, idx), tempBB);
                Value *box;
                if (type_is_ghost(t)) {
                    box = literal_pointer_val(ctx, jt->instance);
                }
                else {
                    jl_cgval_t vinfo_r = jl_cgval_t(vinfo, (jl_value_t*)jt, NULL);
                    box = _boxed_special(ctx, vinfo_r, t);
                    if (!box) {
                        box = emit_allocobj(ctx, jl_datatype_size(jt), literal_pointer_val(ctx, (jl_value_t*)jt));
                        init_bits_cgval(ctx, box, vinfo_r, jl_is_mutable(jt) ? tbaa_mutab : tbaa_immut);
                    }
                }
                box_merge->addIncoming(maybe_decay_untracked(box), tempBB);
                ctx.builder.CreateBr(postBB);
            },
            vinfo.typ,
            counter);
    ctx.builder.SetInsertPoint(defaultBB);
    if (skip.size() > 0 && skip[0]) {
        // skip[0] specifies where to return NULL or the original pointer
        // if the value was not handled above
        box_merge->addIncoming(maybe_decay_untracked(V_null), defaultBB);
        ctx.builder.CreateBr(postBB);
    }
    else if ((vinfo.V == NULL || isa<AllocaInst>(vinfo.V)) && !vinfo.gcroot) {
        Function *trap_func = Intrinsic::getDeclaration(
                ctx.f->getParent(),
                Intrinsic::trap);
        ctx.builder.CreateCall(trap_func);
        ctx.builder.CreateUnreachable();
    }
    else {
        // We're guaranteed here that Load(.gcroot) == .V, because we have determined
        // that this union is a boxed value, rather than an interior pointer of some sort
        box_merge->addIncoming(ctx.builder.CreateLoad(vinfo.gcroot), defaultBB);
        ctx.builder.CreateBr(postBB);
    }
    ctx.builder.SetInsertPoint(postBB);
    return box_merge;
}

// this is used to wrap values for generic contexts, where a
// dynamically-typed value is required (e.g. argument to unknown function).
// if it's already a pointer it's left alone.
static Value *boxed(jl_codectx_t &ctx, const jl_cgval_t &vinfo, bool gcrooted)
{
    jl_value_t *jt = vinfo.typ;
    if (jt == jl_bottom_type || jt == NULL)
        // We have an undef value on a (hopefully) dead branch
        return UndefValue::get(T_prjlvalue);
    if (vinfo.constant)
        return literal_pointer_val(ctx, vinfo.constant);
    if (vinfo.isboxed) {
        assert(vinfo.V && "Missing value for box.");
        // We're guaranteed here that Load(.gcroot) == .V, because we have determined
        // that this value is a box, so if it has a gcroot, that's where the value is.
        return vinfo.gcroot ? ctx.builder.CreateLoad(vinfo.gcroot) : vinfo.V;
    }

    Value *box;
    if (vinfo.TIndex) {
        SmallBitVector skip_none;
        box = box_union(ctx, vinfo, skip_none);
    }
    else {
        assert(vinfo.V && "Missing data for unboxed value.");
        assert(jl_isbits(jt) && jl_is_leaf_type(jt) && "This type shouldn't have been unboxed.");
        Type *t = julia_type_to_llvm(jt);
        assert(!type_is_ghost(t)); // ghost values should have been handled by vinfo.constant above!
        box = _boxed_special(ctx, vinfo, t);
        if (!box) {
            box = emit_allocobj(ctx, jl_datatype_size(jt), literal_pointer_val(ctx, (jl_value_t*)jt));
            init_bits_cgval(ctx, box, vinfo, jl_is_mutable(jt) ? tbaa_mutab : tbaa_immut);
        }
    }
    if (gcrooted) {
        // make a gcroot for the new box
        // (unless the caller explicitly said this was unnecessary)
        Value *froot = emit_local_root(ctx);
        ctx.builder.CreateStore(box, froot);
    }
    return box;
}

// copy src to dest, if src is isbits. if skip is true, the value of dest is undefined
static void emit_unionmove(jl_codectx_t &ctx, Value *dest, const jl_cgval_t &src, Value *skip, bool isVolatile, MDNode *tbaa)
{
    if (AllocaInst *ai = dyn_cast<AllocaInst>(dest))
        ctx.builder.CreateStore(UndefValue::get(ai->getAllocatedType()), ai);
    if (jl_is_leaf_type(src.typ) || src.constant) {
        jl_value_t *typ = src.constant ? jl_typeof(src.constant) : src.typ;
        Type *store_ty = julia_type_to_llvm(typ);
        assert(skip || jl_isbits(typ));
        if (jl_isbits(typ) && jl_datatype_size(typ) > 0) {
            if (!src.ispointer() || src.constant) {
                emit_unbox(ctx, store_ty, src, typ, dest, isVolatile);
            }
            else {
                Value *src_ptr = data_pointer(ctx, src, T_pint8);
                if (dest->getType() != T_pint8)
                    dest = emit_bitcast(ctx, dest, T_pint8);
                if (skip) // copy dest -> dest to simulate an undef value / conditional copy
                    src_ptr = ctx.builder.CreateSelect(skip, dest, src_ptr);
                unsigned nb = jl_datatype_size(typ);
                unsigned alignment = 0;
                ctx.builder.CreateMemCpy(dest, src_ptr, nb, alignment, isVolatile, tbaa);
            }
        }
    }
    else if (src.TIndex) {
        Value *tindex = ctx.builder.CreateAnd(src.TIndex, ConstantInt::get(T_int8, 0x7f));
        if (skip)
            tindex = ctx.builder.CreateSelect(skip, ConstantInt::get(T_int8, 0), tindex);
        Value *src_ptr = data_pointer(ctx, src, T_pint8);
        if (dest->getType() != T_pint8)
            dest = emit_bitcast(ctx, dest, T_pint8);
        BasicBlock *defaultBB = BasicBlock::Create(jl_LLVMContext, "union_move_skip", ctx.f);
        SwitchInst *switchInst = ctx.builder.CreateSwitch(tindex, defaultBB);
        BasicBlock *postBB = BasicBlock::Create(jl_LLVMContext, "post_union_move", ctx.f);
        unsigned counter = 0;
        bool allunboxed = for_each_uniontype_small(
                [&](unsigned idx, jl_datatype_t *jt) {
                    unsigned nb = jl_datatype_size(jt);
                    unsigned alignment = 0;
                    BasicBlock *tempBB = BasicBlock::Create(jl_LLVMContext, "union_move", ctx.f);
                    ctx.builder.SetInsertPoint(tempBB);
                    switchInst->addCase(ConstantInt::get(T_int8, idx), tempBB);
                    if (nb > 0)
                        ctx.builder.CreateMemCpy(dest, src_ptr, nb, alignment, isVolatile, tbaa);
                    ctx.builder.CreateBr(postBB);
                },
                src.typ,
                counter);
        ctx.builder.SetInsertPoint(defaultBB);
        if (!skip && allunboxed && (src.V == NULL || isa<AllocaInst>(src.V))) {
            Function *trap_func = Intrinsic::getDeclaration(
                    ctx.f->getParent(),
                    Intrinsic::trap);
            ctx.builder.CreateCall(trap_func);
            ctx.builder.CreateUnreachable();
        }
        else {
            ctx.builder.CreateBr(postBB);
        }
        ctx.builder.SetInsertPoint(postBB);
    }
    else {
        Value *datatype = emit_typeof_boxed(ctx, src);
        Value *copy_bytes = emit_datatype_size(ctx, datatype);
        if (skip)
            copy_bytes = ctx.builder.CreateSelect(skip, ConstantInt::get(copy_bytes->getType(), 0), copy_bytes);
        ctx.builder.CreateMemCpy(dest,
                             data_pointer(ctx, src, T_pint8),
                             copy_bytes,
                             /*TODO: min-align*/1);
    }
}


static void emit_cpointercheck(jl_codectx_t &ctx, const jl_cgval_t &x, const std::string &msg)
{
    Value *t = emit_typeof_boxed(ctx, x);
    emit_typecheck(ctx, mark_julia_type(ctx, t, true, jl_any_type, false), (jl_value_t*)jl_datatype_type, msg);

    Value *istype =
        ctx.builder.CreateICmpEQ(mark_callee_rooted(emit_datatype_name(ctx, t)),
                             mark_callee_rooted(literal_pointer_val(ctx, (jl_value_t*)jl_pointer_typename)));
    BasicBlock *failBB = BasicBlock::Create(jl_LLVMContext,"fail",ctx.f);
    BasicBlock *passBB = BasicBlock::Create(jl_LLVMContext,"pass");
    ctx.builder.CreateCondBr(istype, passBB, failBB);
    ctx.builder.SetInsertPoint(failBB);

    emit_type_error(ctx, x, literal_pointer_val(ctx, (jl_value_t*)jl_pointer_type), msg);
    ctx.builder.CreateUnreachable();

    ctx.f->getBasicBlockList().push_back(passBB);
    ctx.builder.SetInsertPoint(passBB);
}

// allocation for known size object
static Value *emit_allocobj(jl_codectx_t &ctx, size_t static_size, Value *jt)
{
    JL_FEAT_REQUIRE(ctx, dynamic_alloc);
    JL_FEAT_REQUIRE(ctx, runtime);

    int osize;
    int offset = jl_gc_classify_pools(static_size, &osize);
    Value *ptls_ptr = emit_bitcast(ctx, ctx.ptlsStates, T_pint8);
    Value *v;
    if (offset < 0) {
        Value *args[] = {ptls_ptr,
                         ConstantInt::get(T_size, static_size + sizeof(void*))};
        v = ctx.builder.CreateCall(prepare_call(jlalloc_big_func),
                               ArrayRef<Value*>(args, 2));
    }
    else {
        Value *pool_offs = ConstantInt::get(T_int32, offset);
        Value *args[] = {ptls_ptr, pool_offs, ConstantInt::get(T_int32, osize)};
        v = ctx.builder.CreateCall(prepare_call(jlalloc_pool_func),
                               ArrayRef<Value*>(args, 3));
    }
    tbaa_decorate(tbaa_tag, ctx.builder.CreateStore(maybe_decay_untracked(jt), emit_typeptr_addr(ctx, v)));
    return v;
}

// if ptr is NULL this emits a write barrier _back_
static void emit_write_barrier(jl_codectx_t &ctx, Value *parent, Value *ptr)
{
    Value *parenttag = emit_bitcast(ctx, emit_typeptr_addr(ctx, parent), T_psize);
    Value *parent_type = tbaa_decorate(tbaa_tag, ctx.builder.CreateLoad(parenttag));
    Value *parent_bits = ctx.builder.CreateAnd(parent_type, 3);

    // the branch hint does not seem to make it to the generated code
    Value *parent_old_marked = ctx.builder.CreateICmpEQ(parent_bits,
                                                    ConstantInt::get(T_size, 3));

    BasicBlock *cont = BasicBlock::Create(jl_LLVMContext, "cont");
    BasicBlock *barrier_may_trigger = BasicBlock::Create(jl_LLVMContext, "wb_may_trigger", ctx.f);
    BasicBlock *barrier_trigger = BasicBlock::Create(jl_LLVMContext, "wb_trigger", ctx.f);
    ctx.builder.CreateCondBr(parent_old_marked, barrier_may_trigger, cont);

    ctx.builder.SetInsertPoint(barrier_may_trigger);
    Value *ptr_mark_bit = ctx.builder.CreateAnd(tbaa_decorate(tbaa_tag,
        ctx.builder.CreateLoad(emit_bitcast(ctx, emit_typeptr_addr(ctx, ptr), T_psize))), 1);
    Value *ptr_not_marked = ctx.builder.CreateICmpEQ(ptr_mark_bit, ConstantInt::get(T_size, 0));
    ctx.builder.CreateCondBr(ptr_not_marked, barrier_trigger, cont);
    ctx.builder.SetInsertPoint(barrier_trigger);
    ctx.builder.CreateCall(prepare_call(queuerootfun), maybe_decay_untracked(emit_bitcast(ctx, parent, T_prjlvalue)));
    ctx.builder.CreateBr(cont);
    ctx.f->getBasicBlockList().push_back(cont);
    ctx.builder.SetInsertPoint(cont);
}

static void emit_checked_write_barrier(jl_codectx_t &ctx, Value *parent, Value *ptr)
{
    BasicBlock *cont;
    Value *not_null = ctx.builder.CreateICmpNE(mark_callee_rooted(ptr), mark_callee_rooted(V_null));
    BasicBlock *if_not_null = BasicBlock::Create(jl_LLVMContext, "wb_not_null", ctx.f);
    cont = BasicBlock::Create(jl_LLVMContext, "cont");
    ctx.builder.CreateCondBr(not_null, if_not_null, cont);
    ctx.builder.SetInsertPoint(if_not_null);
    emit_write_barrier(ctx, parent, ptr);
    ctx.builder.CreateBr(cont);
    ctx.f->getBasicBlockList().push_back(cont);
    ctx.builder.SetInsertPoint(cont);
}

static void emit_setfield(jl_codectx_t &ctx,
        jl_datatype_t *sty, const jl_cgval_t &strct, size_t idx0,
        const jl_cgval_t &rhs, bool checked, bool wb)
{
    if (sty->mutabl || !checked) {
        assert(strct.ispointer());
        Value *addr = ctx.builder.CreateGEP(data_pointer(ctx, strct, T_pint8),
                ConstantInt::get(T_size, jl_field_offset(sty, idx0)));
        jl_value_t *jfty = jl_svecref(sty->types, idx0);
        if (jl_field_isptr(sty, idx0)) {
            Value *r = maybe_decay_untracked(boxed(ctx, rhs, false)); // don't need a temporary gcroot since it'll be rooted by strct
            tbaa_decorate(strct.tbaa, ctx.builder.CreateStore(r,
                emit_bitcast(ctx, addr, T_pprjlvalue)));
            if (wb && strct.isboxed)
                emit_checked_write_barrier(ctx, boxed(ctx, strct), r);
        }
        else {
            if (jl_is_uniontype(jfty)) {
                int fsz = jl_field_size(sty, idx0);
                // compute tindex from rhs
                jl_cgval_t rhs_union = convert_julia_type(rhs, jfty, ctx);
                Value *ptindex = builder.CreateGEP(LLVM37_param(T_int8) emit_bitcast(addr, T_pint8), ConstantInt::get(T_size, fsz - 1));
                Value *tindex = compute_tindex_unboxed(rhs_union, jfty, ctx);
                tindex = builder.CreateNUWSub(tindex, ConstantInt::get(T_int8, 1));
                builder.CreateStore(tindex, ptindex);
                // copy data
                emit_unionmove(addr, rhs, NULL, false, NULL, ctx);
            }
            else {
                int align = jl_field_offset(sty, idx0);
                align |= 16;
                align &= -align;
                typed_store(addr, ConstantInt::get(T_size, 0), rhs, jfty, ctx,
                    strct.tbaa, data_pointer(ctx, strct, T_pjlvalue), align);
            }
        }
    }
    else {
        // TODO: better error
        emit_error(ctx, "type is immutable");
    }
}

static bool might_need_root(jl_value_t *ex)
{
    return (!jl_is_symbol(ex) && !jl_is_slot(ex) && !jl_is_ssavalue(ex) &&
            !jl_is_bool(ex) && !jl_is_quotenode(ex) && !jl_is_string(ex) &&
            !jl_is_globalref(ex));
}

static jl_cgval_t emit_new_struct(jl_codectx_t &ctx, jl_value_t *ty, size_t nargs, jl_value_t **args)
{
    assert(jl_is_datatype(ty));
    assert(jl_is_leaf_type(ty));
    assert(nargs>0);
    jl_datatype_t *sty = (jl_datatype_t*)ty;
    size_t nf = jl_datatype_nfields(sty);
    if (nf > 0) {
        if (jl_isbits(sty)) {
            Type *lt = julia_type_to_llvm(ty);
            // whether we should perform the initialization with the struct as a IR value
            // or instead initialize the stack buffer with stores
            bool init_as_value = false;
            if (lt->isVectorTy() ||
                jl_is_vecelement_type(ty) ||
                type_is_ghost(lt)) // maybe also check the size ?
                init_as_value = true;

            size_t na = nargs-1 < nf ? nargs-1 : nf;
            Value *strct;
            if (init_as_value)
                strct = UndefValue::get(lt == T_void ? NoopType : lt);
            else
                strct = emit_static_alloca(ctx, lt);

            unsigned idx = 0;
            for (size_t i = 0; i < na; i++) {
                jl_value_t *jtype = jl_svecref(sty->types, i);
                Type *fty = julia_type_to_llvm(jtype);
                jl_cgval_t fval_info = emit_expr(ctx, args[i + 1]);
                emit_typecheck(ctx, fval_info, jtype, "new");
                if (!type_is_ghost(fty)) {
                    Value *fval = NULL, *dest = NULL;
                    if (!init_as_value) {
                        // avoid unboxing the argument explicitely
                        // and use memcpy instead
                        dest = ctx.builder.CreateConstInBoundsGEP2_32(lt, strct, 0, i);
                    }
                    fval = emit_unbox(ctx, fty, fval_info, jtype, dest);

                    if (init_as_value) {
                        if (lt->isVectorTy())
                            strct = ctx.builder.CreateInsertElement(strct, fval, ConstantInt::get(T_int32,idx));
                        else if (jl_is_vecelement_type(ty))
                            strct = fval;  // VecElement type comes unwrapped in LLVM.
                        else if (lt->isAggregateType())
                            strct = ctx.builder.CreateInsertValue(strct, fval, ArrayRef<unsigned>(&idx,1));
                        else
                            assert(false);
                    }
                }
                idx++;
            }
            if (init_as_value)
                return mark_julia_type(ctx, strct, false, ty);
            else
                return mark_julia_slot(strct, ty, NULL, tbaa_stack);
        }
        Value *strct = emit_allocobj(ctx, jl_datatype_size(sty),
                                     literal_pointer_val(ctx, (jl_value_t*)ty));
        jl_cgval_t strctinfo = mark_julia_type(ctx, strct, true, ty);
        for (size_t i = 0; i < nf; i++) {
            if (jl_field_isptr(sty, i)) {
                tbaa_decorate(strctinfo.tbaa, ctx.builder.CreateStore(
                        ConstantPointerNull::get(cast<PointerType>(T_prjlvalue)),
                        emit_bitcast(ctx,
                            ctx.builder.CreateGEP(emit_bitcast(ctx, decay_derived(strct), T_pint8),
                                ConstantInt::get(T_size, jl_field_offset(sty, i))),
                            T_pprjlvalue)));
            }
        }
        bool need_wb = false;
        // TODO: verify that nargs <= nf (currently handled by front-end)
        for (size_t i = 1; i < nargs; i++) {
            jl_cgval_t rhs = emit_expr(ctx, args[i]);
            if (jl_field_isptr(sty, i - 1) && !rhs.isboxed) {
                need_wb = true;
            }
            if (rhs.isboxed) {
                emit_typecheck(ctx, rhs, jl_svecref(sty->types, i - 1), "new");
            }
            if (might_need_root(args[i])) // TODO: how to remove this?
                need_wb = true;
            emit_setfield(ctx, sty, strctinfo, i - 1, rhs, false, need_wb);
        }
        return strctinfo;
    }
    else if (!sty->mutabl) {
        // 0 fields, ghost or bitstype
        if (jl_datatype_nbits(sty) == 0)
            return ghostValue(sty);
        if (nargs >= 2)
            return emit_expr(ctx, args[1]);  // do side effects
        bool isboxed;
        Type *lt = julia_type_to_llvm(ty, &isboxed);
        assert(!isboxed);
        return mark_julia_type(ctx, UndefValue::get(lt), false, ty);
    }
    else {
        // 0 fields, singleton
        assert(sty->instance != NULL);
        return mark_julia_const(sty->instance);
    }
}

static Value *emit_exc_in_transit(jl_codectx_t &ctx)
{
    Value *pexc_in_transit = emit_bitcast(ctx, ctx.ptlsStates, T_pprjlvalue);
    Constant *offset = ConstantInt::getSigned(T_int32,
        offsetof(jl_tls_states_t, exception_in_transit) / sizeof(void*));
    return ctx.builder.CreateGEP(pexc_in_transit, ArrayRef<Value*>(offset), "jl_exception_in_transit");
}

static void emit_signal_fence(jl_codectx_t &ctx)
{
#if defined(_CPU_ARM_) || defined(_CPU_AARCH64_)
    // LLVM generates very inefficient code (and might include function call)
    // for signal fence. Fallback to the poor man signal fence with
    // inline asm instead.
    // https://llvm.org/bugs/show_bug.cgi?id=27545
    ctx.builder.CreateCall(InlineAsm::get(FunctionType::get(T_void, false), "",
                                      "~{memory}", true));
#else
    ctx.builder.CreateFence(AtomicOrdering::SequentiallyConsistent, SingleThread);
#endif
}

static Value *emit_defer_signal(jl_codectx_t &ctx)
{
    Value *ptls = emit_bitcast(ctx, ctx.ptlsStates,
                                        PointerType::get(T_sigatomic, 0));
    Constant *offset = ConstantInt::getSigned(T_int32,
        offsetof(jl_tls_states_t, defer_signal) / sizeof(sig_atomic_t));
    return ctx.builder.CreateGEP(ptls, ArrayRef<Value*>(offset), "jl_defer_signal");
}

static int compare_cgparams(const jl_cgparams_t *a, const jl_cgparams_t *b)
{
    return (a->cached == b->cached) &&
           // language features
           (a->runtime == b->runtime) &&
           (a->exceptions == b->exceptions) &&
           (a->track_allocations == b->track_allocations) &&
           (a->code_coverage == b->code_coverage) &&
           (a->static_alloc == b->static_alloc) &&
           (a->dynamic_alloc == b->dynamic_alloc) &&
           // hooks
           (a->hooks.module_setup == b->hooks.module_setup) &&
           (a->hooks.module_activation == b->hooks.module_activation) &&
           (a->hooks.raise_exception == b->hooks.raise_exception);
}
