#include "eblas.h"
#include "lapacke.h"
#include "string.h"
#include <complex.h>
#include "string.h"
#include "tables.h"
#include "errors.h"

//Various utility functions
//--------------------------------

int translate(ErlNifEnv* env, const ERL_NIF_TERM* terms, const etypes* format, ...){
    va_list valist;
    va_start(valist, format);
    int valid = 1;
    int* i_dest;

    for(int curr=0; format[curr] != e_end; curr++){
        switch(format[curr]){
            
            case e_int:
                valid = enif_get_int(env, terms[curr], va_arg(valist, int*));
            break;

            case e_char:
                char* c_dest = va_arg(valist, char*);
                char* buff = "0";
                enif_get_atom(env, terms[curr], buff, 2, ERL_NIF_LATIN1);
                c_dest[0] = buff[0];
            break;
            
            case e_double:
                valid = enif_get_double(env, terms[curr], va_arg(valist, double*));
            break;

            case e_ptr:
                valid = get_c_binary(env, terms[curr], va_arg(valist, c_binary*));
            break;
            case e_cste_ptr:
                valid = get_cste_binary(env, terms[curr], va_arg(valist, cste_c_binary*));
            break;

            case e_layout:
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomRowMajor)){ *i_dest = CblasRowMajor;}
                else if (enif_is_identical(terms[curr], atomColMajor)) *i_dest = CblasColMajor;
                else valid = 0;
            break;

            case e_transpose: 
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomNoTrans))  *i_dest = CblasNoTrans;
                else if (enif_is_identical(terms[curr], atomTrans))    *i_dest = CblasTrans;
                else if (enif_is_identical(terms[curr], atomConjTrans))*i_dest = CblasConjTrans;
                else valid = 0;
            break;

            case e_uplo: 
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomUpper)) *i_dest = CblasUpper;
                else if (enif_is_identical(terms[curr], atomLower)) *i_dest = CblasLower;
                else valid = 0;
            break;

            case e_diag: 
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomNonUnit)) *i_dest = CblasNonUnit;
                else if (enif_is_identical(terms[curr], atomUnit))    *i_dest = CblasUnit;
                else valid = 0;
                break;

            case e_side:
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomLeft))  *i_dest = CblasLeft;
                else if (enif_is_identical(terms[curr], atomRight)) *i_dest = CblasRight;
                else valid = 0;
                break;

            default:
                valid = 0;
            break;
        }

        if(!valid){
            va_end(valist);
            return curr + 1;
        }
    }
    
    va_end(valist);
    return 0;
}

// C_binary definitions/functions

//Used for debug purpose.
//Likely thread unsafe.
//Usage: debug_write("A double: %lf, an int:%d", double_val, int_val);
int debug_write(const char* fmt, ...){
    FILE* fp = fopen("priv/debug.txt", "a");
    va_list args;

    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);

    fclose(fp);
    return 1;
}

int get_c_binary(ErlNifEnv* env, const ERL_NIF_TERM term, c_binary* result){
    int arity;
    const ERL_NIF_TERM* terms;
    void* resource;

    // Return false if: incorect record size, resource size does not match
    int b =  enif_get_tuple(env, term, &arity, &terms)
                && arity == 4
                && enif_get_uint(env, terms[1], &result->size)
                && enif_get_uint(env, terms[2], &result->offset)
                && enif_get_resource(env, terms[3], c_binary_resource, &resource)
                && result->size == enif_sizeof_resource(resource);

    result->ptr = (unsigned char*) resource;
    
    return b;
}

// *UnIVeRSAl pointer to an erlang type; currently, binary/double/c_binarry.
int get_cste_binary(ErlNifEnv* env, const ERL_NIF_TERM term, cste_c_binary* result){
    if(enif_is_binary(env, term)){
        // Read a binary.
        ErlNifBinary ebin;
        if(!enif_inspect_binary(env, term, &ebin))
            return 0;
        
        result->size    = ebin.size;
        result->offset  = 0;
        result->ptr     = ebin.data;
        result->type    = e_cste_ptr;
    }
    else{
        // Read a cbin.
        c_binary cbin;
        if(get_c_binary(env, term, &cbin)){
            result->size    = cbin.size;
            result->offset  = cbin.offset;
            result->ptr     = (const unsigned char*) cbin.ptr;
            result->type    = e_ptr;
        }
        else{
            // Read a double.
            if(!enif_get_double(env, term, &result->tmp))
                return 0;

            result->size    = 8;
            result->offset  = 0;
            result->ptr     = (const unsigned char*) &result->tmp;
            result->type    = e_double;
        }
    }
    return 1;
}

double get_cste_double(cste_c_binary cb){
    const void* ptr = get_cste_ptr(cb);
    return *(double*) ptr;
}

float get_cste_float(cste_c_binary cb){
    const void* ptr = get_cste_ptr(cb);
    if(cb.type == e_double){
        double val = get_cste_double(cb);
        return (float) val;
    }
    return *(float*) ptr;
}


int in_bounds(int elem_size, int n_elem, int inc, c_binary b){
    int end_offset = b.offset + (elem_size*n_elem*inc);
    return (elem_size > 0 && end_offset >= 0 && end_offset <= b.size)? ERROR_NONE:ERROR_SIGSEV;
}

int in_cste_bounds(int elem_size, int n_elem, int inc, cste_c_binary b){
    int end_offset = b.offset + (elem_size*n_elem*inc);
    return (elem_size > 0 && end_offset >= 0 && end_offset <= b.size)?ERROR_NONE:ERROR_SIGSEV;
}

void set_cste_c_binary(cste_c_binary *ccb, etypes type, unsigned char* ptr){
    //e_int, e_double, e_float_complex, e_double_complex, 
    switch(type){
        case e_int:            ccb->size = sizeof(int);        break;
        case e_double:         ccb->size = sizeof(double);     break;
        case e_float_complex:  ccb->size = sizeof(float)*2;    break;
        case e_double_complex: ccb->size = sizeof(double)*2;   break;
        default:               ccb->size = 0;                  break;
    }

    ccb->type   = type;
    ccb->offset = 0;
    ccb->ptr    = ptr;
}

ERL_NIF_TERM cste_c_binary_to_term(ErlNifEnv* env, cste_c_binary ccb){
    ERL_NIF_TERM result = -1;

    switch(ccb.type){
        case e_int:     int    vali = *(int*)    ccb.ptr; result = enif_make_int(env, vali);    break;
        case e_double:  double vald = *(double*) ccb.ptr; result = enif_make_double(env, vald); break;

        case e_float_complex:
        case e_double_complex:
             ErlNifBinary bin;

            if(enif_alloc_binary(ccb.size, &bin)){
                memcpy(bin.data, ccb.ptr, ccb.size);
                if(!(result = enif_make_binary(env, &bin))){
                    enif_release_binary(&bin);
                    result = enif_make_badarg(env);
                }
            }
        break;

        default:
            result = enif_make_badarg(env);
        break;
    }
    return result;
}



ERL_NIF_TERM new(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    int size = 0;
    if(!enif_get_int(env, argv[0], &size)) return enif_make_badarg(env);

    void* ptr = enif_alloc_resource(c_binary_resource, size);
    ERL_NIF_TERM resource = enif_make_resource(env, ptr);
    enif_release_resource(ptr);
    return resource; 
}

ERL_NIF_TERM copy(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    ErlNifBinary bin;
    c_binary cbin;

    if(!enif_inspect_binary(env, argv[0], &bin)|| !get_c_binary(env, argv[1], &cbin))
         return enif_make_badarg(env);

    memcpy(cbin.ptr + cbin.offset, bin.data, bin.size);

    return enif_make_atom(env, "ok");
}

ERL_NIF_TERM to_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    c_binary cbin;
    ErlNifBinary bin;
    unsigned size;

    if(!enif_get_uint(env, argv[0], &size)
        || !get_c_binary(env, argv[1], &cbin)
        || !enif_alloc_binary(size, &bin))
        return enif_make_badarg(env);

    memcpy(bin.data, cbin.ptr + cbin.offset, size);

    return enif_make_binary(env, &bin);
}

// UNWRAPPER
// https://stackoverflow.com/questions/7666509/hash-function-for-string
unsigned long hash(char *str){
    unsigned long h = 5381;
    int c;

    while ((c = *str++))
        h = ((h << 5) + h) + c; /* hash * 33 + c */

    return h;
}



ERL_NIF_TERM unwrapper(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    int narg;
    const ERL_NIF_TERM* elements;
    char name[20];

    if(!enif_get_tuple(env, *argv, &narg, &elements)
        || !enif_get_atom(env, elements[0], name, 20, ERL_NIF_LATIN1)
    ){
        return enif_make_badarg(env);
    }

    
    int error = ERROR_NONE;
    narg--;
    elements++;

    unsigned long hash_name = hash(name);

    ERL_NIF_TERM result = 0;

    switch(hash_name){


        case sdsdot: {
            int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_end}, &N, &alpha, &X, &incX, &Y, &incY))
            ){
                cblas_sdsdot(N, get_cste_float(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY);
            }
        } break;
        case dsdot: {
            int N; cste_c_binary X; int incX; cste_c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_dsdot(N, get_cste_ptr(X), incX, get_cste_ptr(Y), incY);
            }
        } break;
        case sdot: {
            int N; cste_c_binary X; int incX; cste_c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_sdot(N, get_cste_ptr(X), incX, get_cste_ptr(Y), incY);
            }
        } break;
        case ddot: {
            int N; cste_c_binary X; int incX; cste_c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_ddot(N, get_cste_ptr(X), incX, get_cste_ptr(Y), incY);
            }
        } break;
        case cdotu: {
            int N; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary dotu;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &N, &X, &incX, &Y, &incY, &dotu))
            ){
                cblas_cdotu_sub(N, get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(dotu));
            }
        } break;
        case cdotc: {
            int N; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary dotc;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &N, &X, &incX, &Y, &incY, &dotc))
            ){
                cblas_cdotc_sub(N, get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(dotc));
            }
        } break;
        case zdotu: {
            int N; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary dotu;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &N, &X, &incX, &Y, &incY, &dotu))
            ){
                cblas_zdotu_sub(N, get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(dotu));
            }
        } break;
        case zdotc: {
            int N; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary dotc;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &N, &X, &incX, &Y, &incY, &dotc))
            ){
                cblas_zdotc_sub(N, get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(dotc));
            }
        } break;
        case snrm2: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_snrm2(N, get_cste_ptr(X), incX);
            }
        } break;
        case sasum: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_sasum(N, get_cste_ptr(X), incX);
            }
        } break;
        case dnrm2: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_dnrm2(N, get_cste_ptr(X), incX);
            }
        } break;
        case dasum: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_dasum(N, get_cste_ptr(X), incX);
            }
        } break;
        case scnrm2: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_scnrm2(N, get_cste_ptr(X), incX);
            }
        } break;
        case scasum: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_scasum(N, get_cste_ptr(X), incX);
            }
        } break;
        case dznrm2: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_dznrm2(N, get_cste_ptr(X), incX);
            }
        } break;
        case dzasum: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_dzasum(N, get_cste_ptr(X), incX);
            }
        } break;
        case isamax: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_isamax(N, get_cste_ptr(X), incX);
            }
        } break;
        case idamax: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_idamax(N, get_cste_ptr(X), incX);
            }
        } break;
        case icamax: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_icamax(N, get_cste_ptr(X), incX);
            }
        } break;
        case izamax: {
            int N; cste_c_binary X; int incX;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &N, &X, &incX))
            ){
                cblas_izamax(N, get_cste_ptr(X), incX);
            }
        } break;
        case sswap: {
            int N; c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_sswap(N, get_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case scopy: {
            int N; cste_c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_scopy(N, get_cste_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case saxpy: {
            int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX, &Y, &incY))
            ){
                cblas_saxpy(N, get_cste_float(alpha), get_cste_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case dswap: {
            int N; c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_dswap(N, get_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case dcopy: {
            int N; cste_c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_dcopy(N, get_cste_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case daxpy: {
            int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX, &Y, &incY))
            ){
                cblas_daxpy(N, get_cste_double(alpha), get_cste_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case cswap: {
            int N; c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_cswap(N, get_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case ccopy: {
            int N; cste_c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_ccopy(N, get_cste_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case caxpy: {
            int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX, &Y, &incY))
            ){
                cblas_caxpy(N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case zswap: {
            int N; c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_zswap(N, get_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case zcopy: {
            int N; cste_c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &N, &X, &incX, &Y, &incY))
            ){
                cblas_zcopy(N, get_cste_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case zaxpy: {
            int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary Y; int incY;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX, &Y, &incY))
            ){
                cblas_zaxpy(N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_ptr(Y), incY);
            }
        } break;
        case srotg: {
            c_binary a; c_binary b; c_binary c; c_binary s;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &a, &b, &c, &s))
            ){
                cblas_srotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s));
            }
        } break;
        case srotmg: {
            c_binary d1; c_binary d2; c_binary b1; cste_c_binary b2; c_binary P;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_ptr, e_ptr, e_ptr, e_cste_ptr, e_ptr, e_end}, &d1, &d2, &b1, &b2, &P))
            ){
                cblas_srotmg(get_ptr(d1), get_ptr(d2), get_ptr(b1), get_cste_float(b2), get_ptr(P));
            }
        } break;
        case srot: {
            int N; c_binary X; int incX; c_binary Y; int incY; cste_c_binary c; cste_c_binary s;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_end}, &N, &X, &incX, &Y, &incY, &c, &s))
            ){
                cblas_srot(N, get_ptr(X), incX, get_ptr(Y), incY, get_cste_float(c), get_cste_float(s));
            }
        } break;
        case srotm: {
            int N; c_binary X; int incX; c_binary Y; int incY; cste_c_binary P;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &N, &X, &incX, &Y, &incY, &P))
            ){
                cblas_srotm(N, get_ptr(X), incX, get_ptr(Y), incY, get_cste_ptr(P));
            }
        } break;
        case drotg: {
            c_binary a; c_binary b; c_binary c; c_binary s;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &a, &b, &c, &s))
            ){
                cblas_drotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s));
            }
        } break;
        case drotmg: {
            c_binary d1; c_binary d2; c_binary b1; cste_c_binary b2; c_binary P;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_ptr, e_ptr, e_ptr, e_cste_ptr, e_ptr, e_end}, &d1, &d2, &b1, &b2, &P))
            ){
                cblas_drotmg(get_ptr(d1), get_ptr(d2), get_ptr(b1), get_cste_double(b2), get_ptr(P));
            }
        } break;
        case drot: {
            int N; c_binary X; int incX; c_binary Y; int incY; cste_c_binary c; cste_c_binary s;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_end}, &N, &X, &incX, &Y, &incY, &c, &s))
            ){
                cblas_drot(N, get_ptr(X), incX, get_ptr(Y), incY, get_cste_double(c), get_cste_double(s));
            }
        } break;
        case drotm: {
            int N; c_binary X; int incX; c_binary Y; int incY; cste_c_binary P;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &N, &X, &incX, &Y, &incY, &P))
            ){
                cblas_drotm(N, get_ptr(X), incX, get_ptr(Y), incY, get_cste_ptr(P));
            }
        } break;
        case sscal: {
            int N; cste_c_binary alpha; c_binary X; int incX;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX))
            ){
                cblas_sscal(N, get_cste_float(alpha), get_ptr(X), incX);
            }
        } break;
        case dscal: {
            int N; cste_c_binary alpha; c_binary X; int incX;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX))
            ){
                cblas_dscal(N, get_cste_double(alpha), get_ptr(X), incX);
            }
        } break;
        case cscal: {
            int N; cste_c_binary alpha; c_binary X; int incX;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX))
            ){
                cblas_cscal(N, get_cste_ptr(alpha), get_ptr(X), incX);
            }
        } break;
        case zscal: {
            int N; cste_c_binary alpha; c_binary X; int incX;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX))
            ){
                cblas_zscal(N, get_cste_ptr(alpha), get_ptr(X), incX);
            }
        } break;
        case csscal: {
            int N; cste_c_binary alpha; c_binary X; int incX;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX))
            ){
                cblas_csscal(N, get_cste_float(alpha), get_ptr(X), incX);
            }
        } break;
        case zdscal: {
            int N; cste_c_binary alpha; c_binary X; int incX;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_ptr, e_int, e_end}, &N, &alpha, &X, &incX))
            ){
                cblas_zdscal(N, get_cste_double(alpha), get_ptr(X), incX);
            }
        } break;
        case sgemv: {
            int order; int TransA; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &TransA, &M, &N, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_sgemv(order, TransA, M, N, get_cste_float(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_float(beta), get_ptr(Y), incY);
            }
        } break;
        case sgbmv: {
            int order; int TransA; int M; int N; int KL; int KU; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_int, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &TransA, &M, &N, &KL, &KU, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_sgbmv(order, TransA, M, N, KL, KU, get_cste_float(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_float(beta), get_ptr(Y), incY);
            }
        } break;
        case strmv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &A, &lda, &X, &incX))
            ){
                cblas_strmv(order, Uplo, TransA, Diag, N, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case stbmv: {
            int order; int Uplo; int TransA; int Diag; int N; int K; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &K, &A, &lda, &X, &incX))
            ){
                cblas_stbmv(order, Uplo, TransA, Diag, N, K, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case stpmv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary Ap; c_binary X; int incX;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &Ap, &X, &incX))
            ){
                cblas_stpmv(order, Uplo, TransA, Diag, N, get_cste_ptr(Ap), get_ptr(X), incX);
            }
        } break;
        case strsv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &A, &lda, &X, &incX))
            ){
                cblas_strsv(order, Uplo, TransA, Diag, N, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case stbsv: {
            int order; int Uplo; int TransA; int Diag; int N; int K; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &K, &A, &lda, &X, &incX))
            ){
                cblas_stbsv(order, Uplo, TransA, Diag, N, K, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case stpsv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary Ap; c_binary X; int incX;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &Ap, &X, &incX))
            ){
                cblas_stpsv(order, Uplo, TransA, Diag, N, get_cste_ptr(Ap), get_ptr(X), incX);
            }
        } break;
        case dgemv: {
            int order; int TransA; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &TransA, &M, &N, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_dgemv(order, TransA, M, N, get_cste_double(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_double(beta), get_ptr(Y), incY);
            }
        } break;
        case dgbmv: {
            int order; int TransA; int M; int N; int KL; int KU; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_int, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &TransA, &M, &N, &KL, &KU, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_dgbmv(order, TransA, M, N, KL, KU, get_cste_double(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_double(beta), get_ptr(Y), incY);
            }
        } break;
        case dtrmv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &A, &lda, &X, &incX))
            ){
                cblas_dtrmv(order, Uplo, TransA, Diag, N, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case dtbmv: {
            int order; int Uplo; int TransA; int Diag; int N; int K; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &K, &A, &lda, &X, &incX))
            ){
                cblas_dtbmv(order, Uplo, TransA, Diag, N, K, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case dtpmv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary Ap; c_binary X; int incX;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &Ap, &X, &incX))
            ){
                cblas_dtpmv(order, Uplo, TransA, Diag, N, get_cste_ptr(Ap), get_ptr(X), incX);
            }
        } break;
        case dtrsv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &A, &lda, &X, &incX))
            ){
                cblas_dtrsv(order, Uplo, TransA, Diag, N, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case dtbsv: {
            int order; int Uplo; int TransA; int Diag; int N; int K; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &K, &A, &lda, &X, &incX))
            ){
                cblas_dtbsv(order, Uplo, TransA, Diag, N, K, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case dtpsv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary Ap; c_binary X; int incX;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &Ap, &X, &incX))
            ){
                cblas_dtpsv(order, Uplo, TransA, Diag, N, get_cste_ptr(Ap), get_ptr(X), incX);
            }
        } break;
        case cgemv: {
            int order; int TransA; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &TransA, &M, &N, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_cgemv(order, TransA, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case cgbmv: {
            int order; int TransA; int M; int N; int KL; int KU; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_int, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &TransA, &M, &N, &KL, &KU, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_cgbmv(order, TransA, M, N, KL, KU, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case ctrmv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &A, &lda, &X, &incX))
            ){
                cblas_ctrmv(order, Uplo, TransA, Diag, N, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case ctbmv: {
            int order; int Uplo; int TransA; int Diag; int N; int K; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &K, &A, &lda, &X, &incX))
            ){
                cblas_ctbmv(order, Uplo, TransA, Diag, N, K, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case ctpmv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary Ap; c_binary X; int incX;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &Ap, &X, &incX))
            ){
                cblas_ctpmv(order, Uplo, TransA, Diag, N, get_cste_ptr(Ap), get_ptr(X), incX);
            }
        } break;
        case ctrsv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &A, &lda, &X, &incX))
            ){
                cblas_ctrsv(order, Uplo, TransA, Diag, N, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case ctbsv: {
            int order; int Uplo; int TransA; int Diag; int N; int K; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &K, &A, &lda, &X, &incX))
            ){
                cblas_ctbsv(order, Uplo, TransA, Diag, N, K, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case ctpsv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary Ap; c_binary X; int incX;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &Ap, &X, &incX))
            ){
                cblas_ctpsv(order, Uplo, TransA, Diag, N, get_cste_ptr(Ap), get_ptr(X), incX);
            }
        } break;
        case zgemv: {
            int order; int TransA; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &TransA, &M, &N, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_zgemv(order, TransA, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case zgbmv: {
            int order; int TransA; int M; int N; int KL; int KU; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_int, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &TransA, &M, &N, &KL, &KU, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_zgbmv(order, TransA, M, N, KL, KU, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case ztrmv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &A, &lda, &X, &incX))
            ){
                cblas_ztrmv(order, Uplo, TransA, Diag, N, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case ztbmv: {
            int order; int Uplo; int TransA; int Diag; int N; int K; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &K, &A, &lda, &X, &incX))
            ){
                cblas_ztbmv(order, Uplo, TransA, Diag, N, K, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case ztpmv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary Ap; c_binary X; int incX;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &Ap, &X, &incX))
            ){
                cblas_ztpmv(order, Uplo, TransA, Diag, N, get_cste_ptr(Ap), get_ptr(X), incX);
            }
        } break;
        case ztrsv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &A, &lda, &X, &incX))
            ){
                cblas_ztrsv(order, Uplo, TransA, Diag, N, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case ztbsv: {
            int order; int Uplo; int TransA; int Diag; int N; int K; cste_c_binary A; int lda; c_binary X; int incX;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &K, &A, &lda, &X, &incX))
            ){
                cblas_ztbsv(order, Uplo, TransA, Diag, N, K, get_cste_ptr(A), lda, get_ptr(X), incX);
            }
        } break;
        case ztpsv: {
            int order; int Uplo; int TransA; int Diag; int N; cste_c_binary Ap; c_binary X; int incX;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &TransA, &Diag, &N, &Ap, &X, &incX))
            ){
                cblas_ztpsv(order, Uplo, TransA, Diag, N, get_cste_ptr(Ap), get_ptr(X), incX);
            }
        } break;
        case ssymv: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_ssymv(order, Uplo, N, get_cste_float(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_float(beta), get_ptr(Y), incY);
            }
        } break;
        case ssbmv: {
            int order; int Uplo; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &K, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_ssbmv(order, Uplo, N, K, get_cste_float(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_float(beta), get_ptr(Y), incY);
            }
        } break;
        case sspmv: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary Ap; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &Ap, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_sspmv(order, Uplo, N, get_cste_float(alpha), get_cste_ptr(Ap), get_cste_ptr(X), incX, get_cste_float(beta), get_ptr(Y), incY);
            }
        } break;
        case sger: {
            int order; int M; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &M, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_sger(order, M, N, get_cste_float(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case ssyr: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary A; int lda;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &A, &lda))
            ){
                cblas_ssyr(order, Uplo, N, get_cste_float(alpha), get_cste_ptr(X), incX, get_ptr(A), lda);
            }
        } break;
        case sspr: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary Ap;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Ap))
            ){
                cblas_sspr(order, Uplo, N, get_cste_float(alpha), get_cste_ptr(X), incX, get_ptr(Ap));
            }
        } break;
        case ssyr2: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_ssyr2(order, Uplo, N, get_cste_float(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case sspr2: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Y, &incY, &A))
            ){
                cblas_sspr2(order, Uplo, N, get_cste_float(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A));
            }
        } break;
        case dsymv: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_dsymv(order, Uplo, N, get_cste_double(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_double(beta), get_ptr(Y), incY);
            }
        } break;
        case dsbmv: {
            int order; int Uplo; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &K, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_dsbmv(order, Uplo, N, K, get_cste_double(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_double(beta), get_ptr(Y), incY);
            }
        } break;
        case dspmv: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary Ap; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &Ap, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_dspmv(order, Uplo, N, get_cste_double(alpha), get_cste_ptr(Ap), get_cste_ptr(X), incX, get_cste_double(beta), get_ptr(Y), incY);
            }
        } break;
        case dger: {
            int order; int M; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &M, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_dger(order, M, N, get_cste_double(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case dsyr: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary A; int lda;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &A, &lda))
            ){
                cblas_dsyr(order, Uplo, N, get_cste_double(alpha), get_cste_ptr(X), incX, get_ptr(A), lda);
            }
        } break;
        case dspr: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary Ap;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Ap))
            ){
                cblas_dspr(order, Uplo, N, get_cste_double(alpha), get_cste_ptr(X), incX, get_ptr(Ap));
            }
        } break;
        case dsyr2: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_dsyr2(order, Uplo, N, get_cste_double(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case dspr2: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Y, &incY, &A))
            ){
                cblas_dspr2(order, Uplo, N, get_cste_double(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A));
            }
        } break;
        case chemv: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_chemv(order, Uplo, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case chbmv: {
            int order; int Uplo; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &K, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_chbmv(order, Uplo, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case chpmv: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary Ap; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &Ap, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_chpmv(order, Uplo, N, get_cste_ptr(alpha), get_cste_ptr(Ap), get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case cgeru: {
            int order; int M; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &M, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_cgeru(order, M, N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case cgerc: {
            int order; int M; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &M, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_cgerc(order, M, N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case cher: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary A; int lda;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &A, &lda))
            ){
                cblas_cher(order, Uplo, N, get_cste_float(alpha), get_cste_ptr(X), incX, get_ptr(A), lda);
            }
        } break;
        case chpr: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary A;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &A))
            ){
                cblas_chpr(order, Uplo, N, get_cste_float(alpha), get_cste_ptr(X), incX, get_ptr(A));
            }
        } break;
        case cher2: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_cher2(order, Uplo, N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case chpr2: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary Ap;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Y, &incY, &Ap))
            ){
                cblas_chpr2(order, Uplo, N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(Ap));
            }
        } break;
        case zhemv: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_zhemv(order, Uplo, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case zhbmv: {
            int order; int Uplo; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &K, &alpha, &A, &lda, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_zhbmv(order, Uplo, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case zhpmv: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary Ap; cste_c_binary X; int incX; cste_c_binary beta; c_binary Y; int incY;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &Ap, &X, &incX, &beta, &Y, &incY))
            ){
                cblas_zhpmv(order, Uplo, N, get_cste_ptr(alpha), get_cste_ptr(Ap), get_cste_ptr(X), incX, get_cste_ptr(beta), get_ptr(Y), incY);
            }
        } break;
        case zgeru: {
            int order; int M; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &M, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_zgeru(order, M, N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case zgerc: {
            int order; int M; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &M, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_zgerc(order, M, N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case zher: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary A; int lda;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &A, &lda))
            ){
                cblas_zher(order, Uplo, N, get_cste_double(alpha), get_cste_ptr(X), incX, get_ptr(A), lda);
            }
        } break;
        case zhpr: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; c_binary A;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &A))
            ){
                cblas_zhpr(order, Uplo, N, get_cste_double(alpha), get_cste_ptr(X), incX, get_ptr(A));
            }
        } break;
        case zher2: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary A; int lda;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Y, &incY, &A, &lda))
            ){
                cblas_zher2(order, Uplo, N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(A), lda);
            }
        } break;
        case zhpr2: {
            int order; int Uplo; int N; cste_c_binary alpha; cste_c_binary X; int incX; cste_c_binary Y; int incY; c_binary Ap;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &order, &Uplo, &N, &alpha, &X, &incX, &Y, &incY, &Ap))
            ){
                cblas_zhpr2(order, Uplo, N, get_cste_ptr(alpha), get_cste_ptr(X), incX, get_cste_ptr(Y), incY, get_ptr(Ap));
            }
        } break;
        case sgemm: {
            int Order; int TransA; int TransB; int M; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_transpose, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &TransA, &TransB, &M, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_sgemm(Order, TransA, TransB, M, N, K, get_cste_float(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_float(beta), get_ptr(C), ldc);
            }
        } break;
        case ssymm: {
            int Order; int Side; int Uplo; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &M, &N, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_ssymm(Order, Side, Uplo, M, N, get_cste_float(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_float(beta), get_ptr(C), ldc);
            }
        } break;
        case ssyrk: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &beta, &C, &ldc))
            ){
                cblas_ssyrk(Order, Uplo, Trans, N, K, get_cste_float(alpha), get_cste_ptr(A), lda, get_cste_float(beta), get_ptr(C), ldc);
            }
        } break;
        case ssyr2k: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_ssyr2k(Order, Uplo, Trans, N, K, get_cste_float(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_float(beta), get_ptr(C), ldc);
            }
        } break;
        case strmm: {
            int Order; int Side; int Uplo; int TransA; int Diag; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; c_binary B; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, &A, &lda, &B, &ldb))
            ){
                cblas_strmm(Order, Side, Uplo, TransA, Diag, M, N, get_cste_float(alpha), get_cste_ptr(A), lda, get_ptr(B), ldb);
            }
        } break;
        case strsm: {
            int Order; int Side; int Uplo; int TransA; int Diag; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; c_binary B; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, &A, &lda, &B, &ldb))
            ){
                cblas_strsm(Order, Side, Uplo, TransA, Diag, M, N, get_cste_float(alpha), get_cste_ptr(A), lda, get_ptr(B), ldb);
            }
        } break;
        case dgemm: {
            int Order; int TransA; int TransB; int M; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_transpose, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &TransA, &TransB, &M, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_dgemm(Order, TransA, TransB, M, N, K, get_cste_double(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_double(beta), get_ptr(C), ldc);
            }
        } break;
        case dsymm: {
            int Order; int Side; int Uplo; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &M, &N, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_dsymm(Order, Side, Uplo, M, N, get_cste_double(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_double(beta), get_ptr(C), ldc);
            }
        } break;
        case dsyrk: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &beta, &C, &ldc))
            ){
                cblas_dsyrk(Order, Uplo, Trans, N, K, get_cste_double(alpha), get_cste_ptr(A), lda, get_cste_double(beta), get_ptr(C), ldc);
            }
        } break;
        case dsyr2k: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_dsyr2k(Order, Uplo, Trans, N, K, get_cste_double(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_double(beta), get_ptr(C), ldc);
            }
        } break;
        case dtrmm: {
            int Order; int Side; int Uplo; int TransA; int Diag; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; c_binary B; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, &A, &lda, &B, &ldb))
            ){
                cblas_dtrmm(Order, Side, Uplo, TransA, Diag, M, N, get_cste_double(alpha), get_cste_ptr(A), lda, get_ptr(B), ldb);
            }
        } break;
        case dtrsm: {
            int Order; int Side; int Uplo; int TransA; int Diag; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; c_binary B; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, &A, &lda, &B, &ldb))
            ){
                cblas_dtrsm(Order, Side, Uplo, TransA, Diag, M, N, get_cste_double(alpha), get_cste_ptr(A), lda, get_ptr(B), ldb);
            }
        } break;
        case cgemm: {
            int Order; int TransA; int TransB; int M; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_transpose, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &TransA, &TransB, &M, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_cgemm(Order, TransA, TransB, M, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case csymm: {
            int Order; int Side; int Uplo; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &M, &N, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_csymm(Order, Side, Uplo, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case csyrk: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &beta, &C, &ldc))
            ){
                cblas_csyrk(Order, Uplo, Trans, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case csyr2k: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_csyr2k(Order, Uplo, Trans, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case ctrmm: {
            int Order; int Side; int Uplo; int TransA; int Diag; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; c_binary B; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, &A, &lda, &B, &ldb))
            ){
                cblas_ctrmm(Order, Side, Uplo, TransA, Diag, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_ptr(B), ldb);
            }
        } break;
        case ctrsm: {
            int Order; int Side; int Uplo; int TransA; int Diag; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; c_binary B; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, &A, &lda, &B, &ldb))
            ){
                cblas_ctrsm(Order, Side, Uplo, TransA, Diag, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_ptr(B), ldb);
            }
        } break;
        case zgemm: {
            int Order; int TransA; int TransB; int M; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_transpose, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &TransA, &TransB, &M, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_zgemm(Order, TransA, TransB, M, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case zsymm: {
            int Order; int Side; int Uplo; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &M, &N, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_zsymm(Order, Side, Uplo, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case zsyrk: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &beta, &C, &ldc))
            ){
                cblas_zsyrk(Order, Uplo, Trans, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case zsyr2k: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_zsyr2k(Order, Uplo, Trans, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case ztrmm: {
            int Order; int Side; int Uplo; int TransA; int Diag; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; c_binary B; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, &A, &lda, &B, &ldb))
            ){
                cblas_ztrmm(Order, Side, Uplo, TransA, Diag, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_ptr(B), ldb);
            }
        } break;
        case ztrsm: {
            int Order; int Side; int Uplo; int TransA; int Diag; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; c_binary B; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &TransA, &Diag, &M, &N, &alpha, &A, &lda, &B, &ldb))
            ){
                cblas_ztrsm(Order, Side, Uplo, TransA, Diag, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_ptr(B), ldb);
            }
        } break;
        case chemm: {
            int Order; int Side; int Uplo; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &M, &N, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_chemm(Order, Side, Uplo, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case cherk: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &beta, &C, &ldc))
            ){
                cblas_cherk(Order, Uplo, Trans, N, K, get_cste_float(alpha), get_cste_ptr(A), lda, get_cste_float(beta), get_ptr(C), ldc);
            }
        } break;
        case cher2k: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_cher2k(Order, Uplo, Trans, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_float(beta), get_ptr(C), ldc);
            }
        } break;
        case zhemm: {
            int Order; int Side; int Uplo; int M; int N; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Side, &Uplo, &M, &N, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_zhemm(Order, Side, Uplo, M, N, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_ptr(beta), get_ptr(C), ldc);
            }
        } break;
        case zherk: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &beta, &C, &ldc))
            ){
                cblas_zherk(Order, Uplo, Trans, N, K, get_cste_double(alpha), get_cste_ptr(A), lda, get_cste_double(beta), get_ptr(C), ldc);
            }
        } break;
        case zher2k: {
            int Order; int Uplo; int Trans; int N; int K; cste_c_binary alpha; cste_c_binary A; int lda; cste_c_binary B; int ldb; cste_c_binary beta; c_binary C; int ldc;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &Order, &Uplo, &Trans, &N, &K, &alpha, &A, &lda, &B, &ldb, &beta, &C, &ldc))
            ){
                cblas_zher2k(Order, Uplo, Trans, N, K, get_cste_ptr(alpha), get_cste_ptr(A), lda, get_cste_ptr(B), ldb, get_cste_double(beta), get_ptr(C), ldc);
            }
        } break;
        case sbdsdc: {
            int matrix_layout; char uplo; char compq; int n; c_binary d; c_binary e; c_binary u; int ldu; c_binary vt; int ldvt; c_binary q; c_binary iq;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &compq, &n, &d, &e, &u, &ldu, &vt, &ldvt, &q, &iq))
            ){
                LAPACKE_sbdsdc(matrix_layout, uplo, compq, n, get_ptr(d), get_ptr(e), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(q), get_ptr(iq));
            }
        } break;
        case dbdsdc: {
            int matrix_layout; char uplo; char compq; int n; c_binary d; c_binary e; c_binary u; int ldu; c_binary vt; int ldvt; c_binary q; c_binary iq;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &compq, &n, &d, &e, &u, &ldu, &vt, &ldvt, &q, &iq))
            ){
                LAPACKE_dbdsdc(matrix_layout, uplo, compq, n, get_ptr(d), get_ptr(e), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(q), get_ptr(iq));
            }
        } break;
        case sbdsqr: {
            int matrix_layout; char uplo; int n; int ncvt; int nru; int ncc; c_binary d; c_binary e; c_binary vt; int ldvt; c_binary u; int ldu; c_binary c; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc))
            ){
                LAPACKE_sbdsqr(matrix_layout, uplo, n, ncvt, nru, ncc, get_ptr(d), get_ptr(e), get_ptr(vt), ldvt, get_ptr(u), ldu, get_ptr(c), ldc);
            }
        } break;
        case dbdsqr: {
            int matrix_layout; char uplo; int n; int ncvt; int nru; int ncc; c_binary d; c_binary e; c_binary vt; int ldvt; c_binary u; int ldu; c_binary c; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc))
            ){
                LAPACKE_dbdsqr(matrix_layout, uplo, n, ncvt, nru, ncc, get_ptr(d), get_ptr(e), get_ptr(vt), ldvt, get_ptr(u), ldu, get_ptr(c), ldc);
            }
        } break;
        case cbdsqr: {
            int matrix_layout; char uplo; int n; int ncvt; int nru; int ncc; c_binary d; c_binary e; c_binary vt; int ldvt; c_binary u; int ldu; c_binary c; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc))
            ){
                LAPACKE_cbdsqr(matrix_layout, uplo, n, ncvt, nru, ncc, get_ptr(d), get_ptr(e), get_ptr(vt), ldvt, get_ptr(u), ldu, get_ptr(c), ldc);
            }
        } break;
        case zbdsqr: {
            int matrix_layout; char uplo; int n; int ncvt; int nru; int ncc; c_binary d; c_binary e; c_binary vt; int ldvt; c_binary u; int ldu; c_binary c; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc))
            ){
                LAPACKE_zbdsqr(matrix_layout, uplo, n, ncvt, nru, ncc, get_ptr(d), get_ptr(e), get_ptr(vt), ldvt, get_ptr(u), ldu, get_ptr(c), ldc);
            }
        } break;
        case sdisna: {
            char job; int m; int n; cste_c_binary d; c_binary sep;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_int, e_cste_ptr, e_ptr, e_end}, &job, &m, &n, &d, &sep))
            ){
                LAPACKE_sdisna(job, m, n, get_cste_ptr(d), get_ptr(sep));
            }
        } break;
        case ddisna: {
            char job; int m; int n; cste_c_binary d; c_binary sep;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_int, e_cste_ptr, e_ptr, e_end}, &job, &m, &n, &d, &sep))
            ){
                LAPACKE_ddisna(job, m, n, get_cste_ptr(d), get_ptr(sep));
            }
        } break;
        case sgbbrd: {
            int matrix_layout; char vect; int m; int n; int ncc; int kl; int ku; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq; c_binary pt; int ldpt; c_binary c; int ldc;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &m, &n, &ncc, &kl, &ku, &ab, &ldab, &d, &e, &q, &ldq, &pt, &ldpt, &c, &ldc))
            ){
                LAPACKE_sgbbrd(matrix_layout, vect, m, n, ncc, kl, ku, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq, get_ptr(pt), ldpt, get_ptr(c), ldc);
            }
        } break;
        case dgbbrd: {
            int matrix_layout; char vect; int m; int n; int ncc; int kl; int ku; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq; c_binary pt; int ldpt; c_binary c; int ldc;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &m, &n, &ncc, &kl, &ku, &ab, &ldab, &d, &e, &q, &ldq, &pt, &ldpt, &c, &ldc))
            ){
                LAPACKE_dgbbrd(matrix_layout, vect, m, n, ncc, kl, ku, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq, get_ptr(pt), ldpt, get_ptr(c), ldc);
            }
        } break;
        case cgbbrd: {
            int matrix_layout; char vect; int m; int n; int ncc; int kl; int ku; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq; c_binary pt; int ldpt; c_binary c; int ldc;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &m, &n, &ncc, &kl, &ku, &ab, &ldab, &d, &e, &q, &ldq, &pt, &ldpt, &c, &ldc))
            ){
                LAPACKE_cgbbrd(matrix_layout, vect, m, n, ncc, kl, ku, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq, get_ptr(pt), ldpt, get_ptr(c), ldc);
            }
        } break;
        case zgbbrd: {
            int matrix_layout; char vect; int m; int n; int ncc; int kl; int ku; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq; c_binary pt; int ldpt; c_binary c; int ldc;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &m, &n, &ncc, &kl, &ku, &ab, &ldab, &d, &e, &q, &ldq, &pt, &ldpt, &c, &ldc))
            ){
                LAPACKE_zgbbrd(matrix_layout, vect, m, n, ncc, kl, ku, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq, get_ptr(pt), ldpt, get_ptr(c), ldc);
            }
        } break;
        case sgbcon: {
            int matrix_layout; char norm; int n; int kl; int ku; cste_c_binary ab; int ldab; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &kl, &ku, &ab, &ldab, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_sgbcon(matrix_layout, norm, n, kl, ku, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case dgbcon: {
            int matrix_layout; char norm; int n; int kl; int ku; cste_c_binary ab; int ldab; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &kl, &ku, &ab, &ldab, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_dgbcon(matrix_layout, norm, n, kl, ku, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case cgbcon: {
            int matrix_layout; char norm; int n; int kl; int ku; cste_c_binary ab; int ldab; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &kl, &ku, &ab, &ldab, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_cgbcon(matrix_layout, norm, n, kl, ku, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zgbcon: {
            int matrix_layout; char norm; int n; int kl; int ku; cste_c_binary ab; int ldab; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &kl, &ku, &ab, &ldab, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zgbcon(matrix_layout, norm, n, kl, ku, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case sgbequ: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_sgbequ(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case dgbequ: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_dgbequ(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case cgbequ: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_cgbequ(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case zgbequ: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_zgbequ(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case sgbequb: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_sgbequb(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case dgbequb: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_dgbequb(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case cgbequb: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_cgbequb(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case zgbequb: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_zgbequb(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case sgbrfs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sgbrfs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dgbrfs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dgbrfs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cgbrfs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cgbrfs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zgbrfs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zgbrfs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case sgbrfsx: {
            int matrix_layout; char trans; char equed; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary r; cste_c_binary c; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 25? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trans, &equed, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &r, &c, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_sgbrfsx(matrix_layout, trans, equed, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(r), get_cste_ptr(c), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case dgbrfsx: {
            int matrix_layout; char trans; char equed; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary r; cste_c_binary c; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 25? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trans, &equed, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &r, &c, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_dgbrfsx(matrix_layout, trans, equed, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(r), get_cste_ptr(c), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case cgbrfsx: {
            int matrix_layout; char trans; char equed; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary r; cste_c_binary c; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 25? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trans, &equed, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &r, &c, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_cgbrfsx(matrix_layout, trans, equed, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(r), get_cste_ptr(c), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case zgbrfsx: {
            int matrix_layout; char trans; char equed; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary r; cste_c_binary c; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 25? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trans, &equed, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &r, &c, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_zgbrfsx(matrix_layout, trans, equed, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(r), get_cste_ptr(c), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case sgbsv: {
            int matrix_layout; int n; int kl; int ku; int nrhs; c_binary ab; int ldab; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgbsv(matrix_layout, n, kl, ku, nrhs, get_ptr(ab), ldab, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case dgbsv: {
            int matrix_layout; int n; int kl; int ku; int nrhs; c_binary ab; int ldab; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgbsv(matrix_layout, n, kl, ku, nrhs, get_ptr(ab), ldab, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case cgbsv: {
            int matrix_layout; int n; int kl; int ku; int nrhs; c_binary ab; int ldab; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgbsv(matrix_layout, n, kl, ku, nrhs, get_ptr(ab), ldab, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zgbsv: {
            int matrix_layout; int n; int kl; int ku; int nrhs; c_binary ab; int ldab; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgbsv(matrix_layout, n, kl, ku, nrhs, get_ptr(ab), ldab, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case sgbtrf: {
            int matrix_layout; int m; int n; int kl; int ku; c_binary ab; int ldab; c_binary ipiv;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &ipiv))
            ){
                LAPACKE_sgbtrf(matrix_layout, m, n, kl, ku, get_ptr(ab), ldab, get_ptr(ipiv));
            }
        } break;
        case dgbtrf: {
            int matrix_layout; int m; int n; int kl; int ku; c_binary ab; int ldab; c_binary ipiv;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &ipiv))
            ){
                LAPACKE_dgbtrf(matrix_layout, m, n, kl, ku, get_ptr(ab), ldab, get_ptr(ipiv));
            }
        } break;
        case cgbtrf: {
            int matrix_layout; int m; int n; int kl; int ku; c_binary ab; int ldab; c_binary ipiv;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &ipiv))
            ){
                LAPACKE_cgbtrf(matrix_layout, m, n, kl, ku, get_ptr(ab), ldab, get_ptr(ipiv));
            }
        } break;
        case zgbtrf: {
            int matrix_layout; int m; int n; int kl; int ku; c_binary ab; int ldab; c_binary ipiv;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &ipiv))
            ){
                LAPACKE_zgbtrf(matrix_layout, m, n, kl, ku, get_ptr(ab), ldab, get_ptr(ipiv));
            }
        } break;
        case sgbtrs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgbtrs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case dgbtrs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgbtrs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case cgbtrs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgbtrs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zgbtrs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgbtrs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case sgebak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary scale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &scale, &m, &v, &ldv))
            ){
                LAPACKE_sgebak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(scale), m, get_ptr(v), ldv);
            }
        } break;
        case dgebak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary scale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &scale, &m, &v, &ldv))
            ){
                LAPACKE_dgebak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(scale), m, get_ptr(v), ldv);
            }
        } break;
        case cgebak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary scale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &scale, &m, &v, &ldv))
            ){
                LAPACKE_cgebak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(scale), m, get_ptr(v), ldv);
            }
        } break;
        case zgebak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary scale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &scale, &m, &v, &ldv))
            ){
                LAPACKE_zgebak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(scale), m, get_ptr(v), ldv);
            }
        } break;
        case sgebal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary ilo; c_binary ihi; c_binary scale;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &ilo, &ihi, &scale))
            ){
                LAPACKE_sgebal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(ilo), get_ptr(ihi), get_ptr(scale));
            }
        } break;
        case dgebal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary ilo; c_binary ihi; c_binary scale;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &ilo, &ihi, &scale))
            ){
                LAPACKE_dgebal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(ilo), get_ptr(ihi), get_ptr(scale));
            }
        } break;
        case cgebal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary ilo; c_binary ihi; c_binary scale;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &ilo, &ihi, &scale))
            ){
                LAPACKE_cgebal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(ilo), get_ptr(ihi), get_ptr(scale));
            }
        } break;
        case zgebal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary ilo; c_binary ihi; c_binary scale;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &ilo, &ihi, &scale))
            ){
                LAPACKE_zgebal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(ilo), get_ptr(ihi), get_ptr(scale));
            }
        } break;
        case sgebrd: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tauq; c_binary taup;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &d, &e, &tauq, &taup))
            ){
                LAPACKE_sgebrd(matrix_layout, m, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tauq), get_ptr(taup));
            }
        } break;
        case dgebrd: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tauq; c_binary taup;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &d, &e, &tauq, &taup))
            ){
                LAPACKE_dgebrd(matrix_layout, m, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tauq), get_ptr(taup));
            }
        } break;
        case cgebrd: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tauq; c_binary taup;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &d, &e, &tauq, &taup))
            ){
                LAPACKE_cgebrd(matrix_layout, m, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tauq), get_ptr(taup));
            }
        } break;
        case zgebrd: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tauq; c_binary taup;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &d, &e, &tauq, &taup))
            ){
                LAPACKE_zgebrd(matrix_layout, m, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tauq), get_ptr(taup));
            }
        } break;
        case sgecon: {
            int matrix_layout; char norm; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_sgecon(matrix_layout, norm, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case dgecon: {
            int matrix_layout; char norm; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_dgecon(matrix_layout, norm, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case cgecon: {
            int matrix_layout; char norm; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_cgecon(matrix_layout, norm, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zgecon: {
            int matrix_layout; char norm; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_zgecon(matrix_layout, norm, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case sgeequ: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_sgeequ(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case dgeequ: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_dgeequ(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case cgeequ: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_cgeequ(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case zgeequ: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_zgeequ(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case sgeequb: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_sgeequb(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case dgeequb: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_dgeequb(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case cgeequb: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_cgeequb(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case zgeequb: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_zgeequb(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        } break;
        case sgeev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary wr; c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_sgeev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(wr), get_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        } break;
        case dgeev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary wr; c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_dgeev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(wr), get_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        } break;
        case cgeev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &w, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_cgeev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        } break;
        case zgeev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &w, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_zgeev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        } break;
        case sgeevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary wr; c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary scale; c_binary abnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv))
            ){
                LAPACKE_sgeevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(wr), get_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(scale), get_ptr(abnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        } break;
        case dgeevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary wr; c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary scale; c_binary abnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv))
            ){
                LAPACKE_dgeevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(wr), get_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(scale), get_ptr(abnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        } break;
        case cgeevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary scale; c_binary abnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &w, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv))
            ){
                LAPACKE_cgeevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(scale), get_ptr(abnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        } break;
        case zgeevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary scale; c_binary abnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &w, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv))
            ){
                LAPACKE_zgeevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(scale), get_ptr(abnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        } break;
        case sgehrd: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_sgehrd(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case dgehrd: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_dgehrd(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case cgehrd: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_cgehrd(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case zgehrd: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_zgehrd(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case sgejsv: {
            int matrix_layout; char joba; char jobu; char jobv; char jobr; char jobt; char jobp; int m; int n; c_binary a; int lda; c_binary sva; c_binary u; int ldu; c_binary v; int ldv; c_binary stat; c_binary istat;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, &a, &lda, &sva, &u, &ldu, &v, &ldv, &stat, &istat))
            ){
                LAPACKE_sgejsv(matrix_layout, joba, jobu, jobv, jobr, jobt, jobp, m, n, get_ptr(a), lda, get_ptr(sva), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(stat), get_ptr(istat));
            }
        } break;
        case dgejsv: {
            int matrix_layout; char joba; char jobu; char jobv; char jobr; char jobt; char jobp; int m; int n; c_binary a; int lda; c_binary sva; c_binary u; int ldu; c_binary v; int ldv; c_binary stat; c_binary istat;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, &a, &lda, &sva, &u, &ldu, &v, &ldv, &stat, &istat))
            ){
                LAPACKE_dgejsv(matrix_layout, joba, jobu, jobv, jobr, jobt, jobp, m, n, get_ptr(a), lda, get_ptr(sva), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(stat), get_ptr(istat));
            }
        } break;
        case sgelqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgelqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case dgelqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgelqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case cgelqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgelqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case zgelqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgelqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case sgels: {
            int matrix_layout; char trans; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &trans, &m, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_sgels(matrix_layout, trans, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case dgels: {
            int matrix_layout; char trans; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &trans, &m, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dgels(matrix_layout, trans, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case cgels: {
            int matrix_layout; char trans; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &trans, &m, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_cgels(matrix_layout, trans, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case zgels: {
            int matrix_layout; char trans; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &trans, &m, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_zgels(matrix_layout, trans, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case sgelsd: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_sgelsd(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case dgelsd: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_dgelsd(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case cgelsd: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_cgelsd(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case zgelsd: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_zgelsd(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case sgelss: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_sgelss(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case dgelss: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_dgelss(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case cgelss: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_cgelss(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case zgelss: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_zgelss(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case sgelsy: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary jpvt; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &jpvt, &rcond, &rank))
            ){
                LAPACKE_sgelsy(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(jpvt), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case dgelsy: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary jpvt; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &jpvt, &rcond, &rank))
            ){
                LAPACKE_dgelsy(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(jpvt), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case cgelsy: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary jpvt; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &jpvt, &rcond, &rank))
            ){
                LAPACKE_cgelsy(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(jpvt), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case zgelsy: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary jpvt; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &jpvt, &rcond, &rank))
            ){
                LAPACKE_zgelsy(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(jpvt), get_cste_double(rcond), get_ptr(rank));
            }
        } break;
        case sgeqlf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgeqlf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case dgeqlf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgeqlf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case cgeqlf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgeqlf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case zgeqlf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgeqlf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case sgeqp3: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_sgeqp3(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        } break;
        case dgeqp3: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_dgeqp3(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        } break;
        case cgeqp3: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_cgeqp3(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        } break;
        case zgeqp3: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_zgeqp3(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        } break;
        case sgeqpf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_sgeqpf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        } break;
        case dgeqpf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_dgeqpf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        } break;
        case cgeqpf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_cgeqpf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        } break;
        case zgeqpf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_zgeqpf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        } break;
        case sgeqrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgeqrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case dgeqrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgeqrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case cgeqrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgeqrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case zgeqrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgeqrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case sgeqrfp: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgeqrfp(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case dgeqrfp: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgeqrfp(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case cgeqrfp: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgeqrfp(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case zgeqrfp: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgeqrfp(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case sgerfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sgerfs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dgerfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dgerfs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cgerfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cgerfs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zgerfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zgerfs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case sgerfsx: {
            int matrix_layout; char trans; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary r; cste_c_binary c; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trans, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &r, &c, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_sgerfsx(matrix_layout, trans, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(r), get_cste_ptr(c), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case dgerfsx: {
            int matrix_layout; char trans; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary r; cste_c_binary c; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trans, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &r, &c, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_dgerfsx(matrix_layout, trans, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(r), get_cste_ptr(c), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case cgerfsx: {
            int matrix_layout; char trans; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary r; cste_c_binary c; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trans, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &r, &c, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_cgerfsx(matrix_layout, trans, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(r), get_cste_ptr(c), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case zgerfsx: {
            int matrix_layout; char trans; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary r; cste_c_binary c; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trans, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &r, &c, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_zgerfsx(matrix_layout, trans, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(r), get_cste_ptr(c), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case sgerqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgerqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case dgerqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgerqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case cgerqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgerqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case zgerqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgerqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case sgesdd: {
            int matrix_layout; char jobz; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt))
            ){
                LAPACKE_sgesdd(matrix_layout, jobz, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt);
            }
        } break;
        case dgesdd: {
            int matrix_layout; char jobz; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt))
            ){
                LAPACKE_dgesdd(matrix_layout, jobz, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt);
            }
        } break;
        case cgesdd: {
            int matrix_layout; char jobz; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt))
            ){
                LAPACKE_cgesdd(matrix_layout, jobz, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt);
            }
        } break;
        case zgesdd: {
            int matrix_layout; char jobz; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt))
            ){
                LAPACKE_zgesdd(matrix_layout, jobz, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt);
            }
        } break;
        case sgesv: {
            int matrix_layout; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgesv(matrix_layout, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case dgesv: {
            int matrix_layout; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgesv(matrix_layout, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case cgesv: {
            int matrix_layout; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgesv(matrix_layout, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zgesv: {
            int matrix_layout; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgesv(matrix_layout, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case sgesvd: {
            int matrix_layout; char jobu; char jobvt; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt; c_binary superb;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &superb))
            ){
                LAPACKE_sgesvd(matrix_layout, jobu, jobvt, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(superb));
            }
        } break;
        case dgesvd: {
            int matrix_layout; char jobu; char jobvt; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt; c_binary superb;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &superb))
            ){
                LAPACKE_dgesvd(matrix_layout, jobu, jobvt, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(superb));
            }
        } break;
        case cgesvd: {
            int matrix_layout; char jobu; char jobvt; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt; c_binary superb;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &superb))
            ){
                LAPACKE_cgesvd(matrix_layout, jobu, jobvt, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(superb));
            }
        } break;
        case zgesvd: {
            int matrix_layout; char jobu; char jobvt; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt; c_binary superb;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &superb))
            ){
                LAPACKE_zgesvd(matrix_layout, jobu, jobvt, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(superb));
            }
        } break;
        case sgesvj: {
            int matrix_layout; char joba; char jobu; char jobv; int m; int n; c_binary a; int lda; c_binary sva; int mv; c_binary v; int ldv; c_binary stat;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &joba, &jobu, &jobv, &m, &n, &a, &lda, &sva, &mv, &v, &ldv, &stat))
            ){
                LAPACKE_sgesvj(matrix_layout, joba, jobu, jobv, m, n, get_ptr(a), lda, get_ptr(sva), mv, get_ptr(v), ldv, get_ptr(stat));
            }
        } break;
        case dgesvj: {
            int matrix_layout; char joba; char jobu; char jobv; int m; int n; c_binary a; int lda; c_binary sva; int mv; c_binary v; int ldv; c_binary stat;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &joba, &jobu, &jobv, &m, &n, &a, &lda, &sva, &mv, &v, &ldv, &stat))
            ){
                LAPACKE_dgesvj(matrix_layout, joba, jobu, jobv, m, n, get_ptr(a), lda, get_ptr(sva), mv, get_ptr(v), ldv, get_ptr(stat));
            }
        } break;
        case sgetrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_sgetrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case dgetrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_dgetrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case cgetrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_cgetrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case zgetrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zgetrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case sgetri: {
            int matrix_layout; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_sgetri(matrix_layout, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case dgetri: {
            int matrix_layout; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_dgetri(matrix_layout, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case cgetri: {
            int matrix_layout; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_cgetri(matrix_layout, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case zgetri: {
            int matrix_layout; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zgetri(matrix_layout, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case sgetrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgetrs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case dgetrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgetrs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case cgetrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgetrs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zgetrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgetrs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case sggbak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary lscale; cste_c_binary rscale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &lscale, &rscale, &m, &v, &ldv))
            ){
                LAPACKE_sggbak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(lscale), get_cste_ptr(rscale), m, get_ptr(v), ldv);
            }
        } break;
        case dggbak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary lscale; cste_c_binary rscale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &lscale, &rscale, &m, &v, &ldv))
            ){
                LAPACKE_dggbak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(lscale), get_cste_ptr(rscale), m, get_ptr(v), ldv);
            }
        } break;
        case cggbak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary lscale; cste_c_binary rscale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &lscale, &rscale, &m, &v, &ldv))
            ){
                LAPACKE_cggbak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(lscale), get_cste_ptr(rscale), m, get_ptr(v), ldv);
            }
        } break;
        case zggbak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary lscale; cste_c_binary rscale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &lscale, &rscale, &m, &v, &ldv))
            ){
                LAPACKE_zggbak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(lscale), get_cste_ptr(rscale), m, get_ptr(v), ldv);
            }
        } break;
        case sggbal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary b; int ldb; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &b, &ldb, &ilo, &ihi, &lscale, &rscale))
            ){
                LAPACKE_sggbal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale));
            }
        } break;
        case dggbal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary b; int ldb; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &b, &ldb, &ilo, &ihi, &lscale, &rscale))
            ){
                LAPACKE_dggbal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale));
            }
        } break;
        case cggbal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary b; int ldb; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &b, &ldb, &ilo, &ihi, &lscale, &rscale))
            ){
                LAPACKE_cggbal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale));
            }
        } break;
        case zggbal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary b; int ldb; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &b, &ldb, &ilo, &ihi, &lscale, &rscale))
            ){
                LAPACKE_zggbal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale));
            }
        } break;
        case sggev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_sggev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        } break;
        case dggev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_dggev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        } break;
        case cggev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &b, &ldb, &alpha, &beta, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_cggev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        } break;
        case zggev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &b, &ldb, &alpha, &beta, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_zggev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        } break;
        case sggevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale; c_binary abnrm; c_binary bbnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 25? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &lscale, &rscale, &abnrm, &bbnrm, &rconde, &rcondv))
            ){
                LAPACKE_sggevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale), get_ptr(abnrm), get_ptr(bbnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        } break;
        case dggevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale; c_binary abnrm; c_binary bbnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 25? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &lscale, &rscale, &abnrm, &bbnrm, &rconde, &rcondv))
            ){
                LAPACKE_dggevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale), get_ptr(abnrm), get_ptr(bbnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        } break;
        case cggevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale; c_binary abnrm; c_binary bbnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &b, &ldb, &alpha, &beta, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &lscale, &rscale, &abnrm, &bbnrm, &rconde, &rcondv))
            ){
                LAPACKE_cggevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale), get_ptr(abnrm), get_ptr(bbnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        } break;
        case zggevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale; c_binary abnrm; c_binary bbnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &b, &ldb, &alpha, &beta, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &lscale, &rscale, &abnrm, &bbnrm, &rconde, &rcondv))
            ){
                LAPACKE_zggevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale), get_ptr(abnrm), get_ptr(bbnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        } break;
        case sggglm: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary b; int ldb; c_binary d; c_binary x; c_binary y;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &b, &ldb, &d, &x, &y))
            ){
                LAPACKE_sggglm(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(d), get_ptr(x), get_ptr(y));
            }
        } break;
        case dggglm: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary b; int ldb; c_binary d; c_binary x; c_binary y;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &b, &ldb, &d, &x, &y))
            ){
                LAPACKE_dggglm(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(d), get_ptr(x), get_ptr(y));
            }
        } break;
        case cggglm: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary b; int ldb; c_binary d; c_binary x; c_binary y;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &b, &ldb, &d, &x, &y))
            ){
                LAPACKE_cggglm(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(d), get_ptr(x), get_ptr(y));
            }
        } break;
        case zggglm: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary b; int ldb; c_binary d; c_binary x; c_binary y;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &b, &ldb, &d, &x, &y))
            ){
                LAPACKE_zggglm(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(d), get_ptr(x), get_ptr(y));
            }
        } break;
        case sgghrd: {
            int matrix_layout; char compq; char compz; int n; int ilo; int ihi; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &compq, &compz, &n, &ilo, &ihi, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_sgghrd(matrix_layout, compq, compz, n, ilo, ihi, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz);
            }
        } break;
        case dgghrd: {
            int matrix_layout; char compq; char compz; int n; int ilo; int ihi; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &compq, &compz, &n, &ilo, &ihi, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_dgghrd(matrix_layout, compq, compz, n, ilo, ihi, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz);
            }
        } break;
        case cgghrd: {
            int matrix_layout; char compq; char compz; int n; int ilo; int ihi; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &compq, &compz, &n, &ilo, &ihi, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_cgghrd(matrix_layout, compq, compz, n, ilo, ihi, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz);
            }
        } break;
        case zgghrd: {
            int matrix_layout; char compq; char compz; int n; int ilo; int ihi; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &compq, &compz, &n, &ilo, &ihi, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_zgghrd(matrix_layout, compq, compz, n, ilo, ihi, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz);
            }
        } break;
        case sgglse: {
            int matrix_layout; int m; int n; int p; c_binary a; int lda; c_binary b; int ldb; c_binary c; c_binary d; c_binary x;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &p, &a, &lda, &b, &ldb, &c, &d, &x))
            ){
                LAPACKE_sgglse(matrix_layout, m, n, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(c), get_ptr(d), get_ptr(x));
            }
        } break;
        case dgglse: {
            int matrix_layout; int m; int n; int p; c_binary a; int lda; c_binary b; int ldb; c_binary c; c_binary d; c_binary x;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &p, &a, &lda, &b, &ldb, &c, &d, &x))
            ){
                LAPACKE_dgglse(matrix_layout, m, n, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(c), get_ptr(d), get_ptr(x));
            }
        } break;
        case cgglse: {
            int matrix_layout; int m; int n; int p; c_binary a; int lda; c_binary b; int ldb; c_binary c; c_binary d; c_binary x;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &p, &a, &lda, &b, &ldb, &c, &d, &x))
            ){
                LAPACKE_cgglse(matrix_layout, m, n, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(c), get_ptr(d), get_ptr(x));
            }
        } break;
        case zgglse: {
            int matrix_layout; int m; int n; int p; c_binary a; int lda; c_binary b; int ldb; c_binary c; c_binary d; c_binary x;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &p, &a, &lda, &b, &ldb, &c, &d, &x))
            ){
                LAPACKE_zgglse(matrix_layout, m, n, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(c), get_ptr(d), get_ptr(x));
            }
        } break;
        case sggqrf: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_sggqrf(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        } break;
        case dggqrf: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_dggqrf(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        } break;
        case cggqrf: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_cggqrf(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        } break;
        case zggqrf: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_zggqrf(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        } break;
        case sggrqf: {
            int matrix_layout; int m; int p; int n; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &p, &n, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_sggrqf(matrix_layout, m, p, n, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        } break;
        case dggrqf: {
            int matrix_layout; int m; int p; int n; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &p, &n, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_dggrqf(matrix_layout, m, p, n, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        } break;
        case cggrqf: {
            int matrix_layout; int m; int p; int n; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &p, &n, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_cggrqf(matrix_layout, m, p, n, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        } break;
        case zggrqf: {
            int matrix_layout; int m; int p; int n; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &p, &n, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_zggrqf(matrix_layout, m, p, n, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        } break;
        case sggsvd: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int n; int p; c_binary k; c_binary l; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary iwork;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &n, &p, &k, &l, &a, &lda, &b, &ldb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &iwork))
            ){
                LAPACKE_sggsvd(matrix_layout, jobu, jobv, jobq, m, n, p, get_ptr(k), get_ptr(l), get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(iwork));
            }
        } break;
        case dggsvd: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int n; int p; c_binary k; c_binary l; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary iwork;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &n, &p, &k, &l, &a, &lda, &b, &ldb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &iwork))
            ){
                LAPACKE_dggsvd(matrix_layout, jobu, jobv, jobq, m, n, p, get_ptr(k), get_ptr(l), get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(iwork));
            }
        } break;
        case cggsvd: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int n; int p; c_binary k; c_binary l; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary iwork;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &n, &p, &k, &l, &a, &lda, &b, &ldb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &iwork))
            ){
                LAPACKE_cggsvd(matrix_layout, jobu, jobv, jobq, m, n, p, get_ptr(k), get_ptr(l), get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(iwork));
            }
        } break;
        case zggsvd: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int n; int p; c_binary k; c_binary l; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary iwork;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &n, &p, &k, &l, &a, &lda, &b, &ldb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &iwork))
            ){
                LAPACKE_zggsvd(matrix_layout, jobu, jobv, jobq, m, n, p, get_ptr(k), get_ptr(l), get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(iwork));
            }
        } break;
        case sggsvp: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary k; c_binary l; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &a, &lda, &b, &ldb, &tola, &tolb, &k, &l, &u, &ldu, &v, &ldv, &q, &ldq))
            ){
                LAPACKE_sggsvp(matrix_layout, jobu, jobv, jobq, m, p, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(k), get_ptr(l), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq);
            }
        } break;
        case dggsvp: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary k; c_binary l; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &a, &lda, &b, &ldb, &tola, &tolb, &k, &l, &u, &ldu, &v, &ldv, &q, &ldq))
            ){
                LAPACKE_dggsvp(matrix_layout, jobu, jobv, jobq, m, p, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(k), get_ptr(l), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq);
            }
        } break;
        case cggsvp: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary k; c_binary l; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &a, &lda, &b, &ldb, &tola, &tolb, &k, &l, &u, &ldu, &v, &ldv, &q, &ldq))
            ){
                LAPACKE_cggsvp(matrix_layout, jobu, jobv, jobq, m, p, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(k), get_ptr(l), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq);
            }
        } break;
        case zggsvp: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary k; c_binary l; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &a, &lda, &b, &ldb, &tola, &tolb, &k, &l, &u, &ldu, &v, &ldv, &q, &ldq))
            ){
                LAPACKE_zggsvp(matrix_layout, jobu, jobv, jobq, m, p, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(k), get_ptr(l), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq);
            }
        } break;
        case sgtcon: {
            char norm; int n; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &norm, &n, &dl, &d, &du, &du2, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_sgtcon(norm, n, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case dgtcon: {
            char norm; int n; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &norm, &n, &dl, &d, &du, &du2, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_dgtcon(norm, n, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case cgtcon: {
            char norm; int n; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &norm, &n, &dl, &d, &du, &du2, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_cgtcon(norm, n, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zgtcon: {
            char norm; int n; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &norm, &n, &dl, &d, &du, &du2, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zgtcon(norm, n, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case sgtrfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary dlf; cste_c_binary df; cste_c_binary duf; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sgtrfs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(dlf), get_cste_ptr(df), get_cste_ptr(duf), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dgtrfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary dlf; cste_c_binary df; cste_c_binary duf; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dgtrfs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(dlf), get_cste_ptr(df), get_cste_ptr(duf), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cgtrfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary dlf; cste_c_binary df; cste_c_binary duf; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cgtrfs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(dlf), get_cste_ptr(df), get_cste_ptr(duf), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zgtrfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary dlf; cste_c_binary df; cste_c_binary duf; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zgtrfs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(dlf), get_cste_ptr(df), get_cste_ptr(duf), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case sgtsv: {
            int matrix_layout; int n; int nrhs; c_binary dl; c_binary d; c_binary du; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &dl, &d, &du, &b, &ldb))
            ){
                LAPACKE_sgtsv(matrix_layout, n, nrhs, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(b), ldb);
            }
        } break;
        case dgtsv: {
            int matrix_layout; int n; int nrhs; c_binary dl; c_binary d; c_binary du; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &dl, &d, &du, &b, &ldb))
            ){
                LAPACKE_dgtsv(matrix_layout, n, nrhs, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(b), ldb);
            }
        } break;
        case cgtsv: {
            int matrix_layout; int n; int nrhs; c_binary dl; c_binary d; c_binary du; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &dl, &d, &du, &b, &ldb))
            ){
                LAPACKE_cgtsv(matrix_layout, n, nrhs, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(b), ldb);
            }
        } break;
        case zgtsv: {
            int matrix_layout; int n; int nrhs; c_binary dl; c_binary d; c_binary du; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &dl, &d, &du, &b, &ldb))
            ){
                LAPACKE_zgtsv(matrix_layout, n, nrhs, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(b), ldb);
            }
        } break;
        case sgtsvx: {
            int matrix_layout; char fact; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; c_binary dlf; c_binary df; c_binary duf; c_binary du2; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_sgtsvx(matrix_layout, fact, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_ptr(dlf), get_ptr(df), get_ptr(duf), get_ptr(du2), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dgtsvx: {
            int matrix_layout; char fact; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; c_binary dlf; c_binary df; c_binary duf; c_binary du2; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_dgtsvx(matrix_layout, fact, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_ptr(dlf), get_ptr(df), get_ptr(duf), get_ptr(du2), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cgtsvx: {
            int matrix_layout; char fact; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; c_binary dlf; c_binary df; c_binary duf; c_binary du2; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_cgtsvx(matrix_layout, fact, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_ptr(dlf), get_ptr(df), get_ptr(duf), get_ptr(du2), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zgtsvx: {
            int matrix_layout; char fact; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; c_binary dlf; c_binary df; c_binary duf; c_binary du2; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zgtsvx(matrix_layout, fact, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_ptr(dlf), get_ptr(df), get_ptr(duf), get_ptr(du2), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case sgttrf: {
            int n; c_binary dl; c_binary d; c_binary du; c_binary du2; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &n, &dl, &d, &du, &du2, &ipiv))
            ){
                LAPACKE_sgttrf(n, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(du2), get_ptr(ipiv));
            }
        } break;
        case dgttrf: {
            int n; c_binary dl; c_binary d; c_binary du; c_binary du2; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &n, &dl, &d, &du, &du2, &ipiv))
            ){
                LAPACKE_dgttrf(n, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(du2), get_ptr(ipiv));
            }
        } break;
        case cgttrf: {
            int n; c_binary dl; c_binary d; c_binary du; c_binary du2; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &n, &dl, &d, &du, &du2, &ipiv))
            ){
                LAPACKE_cgttrf(n, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(du2), get_ptr(ipiv));
            }
        } break;
        case zgttrf: {
            int n; c_binary dl; c_binary d; c_binary du; c_binary du2; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &n, &dl, &d, &du, &du2, &ipiv))
            ){
                LAPACKE_zgttrf(n, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(du2), get_ptr(ipiv));
            }
        } break;
        case sgttrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &du2, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgttrs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case dgttrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &du2, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgttrs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case cgttrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &du2, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgttrs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zgttrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &du2, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgttrs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case chbev: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_chbev(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case zhbev: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_zhbev(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case chbevd: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_chbevd(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case zhbevd: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_zhbevd(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case chbevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int kd; c_binary ab; int ldab; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &kd, &ab, &ldab, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chbevx(matrix_layout, jobz, range, uplo, n, kd, get_ptr(ab), ldab, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case zhbevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int kd; c_binary ab; int ldab; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &kd, &ab, &ldab, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhbevx(matrix_layout, jobz, range, uplo, n, kd, get_ptr(ab), ldab, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case chbgst: {
            int matrix_layout; char vect; char uplo; int n; int ka; int kb; c_binary ab; int ldab; cste_c_binary bb; int ldbb; c_binary x; int ldx;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &x, &ldx))
            ){
                LAPACKE_chbgst(matrix_layout, vect, uplo, n, ka, kb, get_ptr(ab), ldab, get_cste_ptr(bb), ldbb, get_ptr(x), ldx);
            }
        } break;
        case zhbgst: {
            int matrix_layout; char vect; char uplo; int n; int ka; int kb; c_binary ab; int ldab; cste_c_binary bb; int ldbb; c_binary x; int ldx;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &x, &ldx))
            ){
                LAPACKE_zhbgst(matrix_layout, vect, uplo, n, ka, kb, get_ptr(ab), ldab, get_cste_ptr(bb), ldbb, get_ptr(x), ldx);
            }
        } break;
        case chbgv: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_chbgv(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case zhbgv: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_zhbgv(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case chbgvd: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_chbgvd(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case zhbgvd: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_zhbgvd(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case chbgvx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chbgvx(matrix_layout, jobz, range, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case zhbgvx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhbgvx(matrix_layout, jobz, range, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case chbtrd: {
            int matrix_layout; char vect; char uplo; int n; int kd; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &kd, &ab, &ldab, &d, &e, &q, &ldq))
            ){
                LAPACKE_chbtrd(matrix_layout, vect, uplo, n, kd, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq);
            }
        } break;
        case zhbtrd: {
            int matrix_layout; char vect; char uplo; int n; int kd; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &kd, &ab, &ldab, &d, &e, &q, &ldq))
            ){
                LAPACKE_zhbtrd(matrix_layout, vect, uplo, n, kd, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq);
            }
        } break;
        case checon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_checon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zhecon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zhecon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case cheequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_cheequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case zheequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_zheequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case cheev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_cheev(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        } break;
        case zheev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_zheev(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        } break;
        case cheevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_cheevd(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        } break;
        case zheevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_zheevd(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        } break;
        case cheevr: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_cheevr(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case zheevr: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_zheevr(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case cheevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_cheevx(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case zheevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zheevx(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case chegst: {
            int matrix_layout; int itype; char uplo; int n; c_binary a; int lda; cste_c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_end}, &matrix_layout, &itype, &uplo, &n, &a, &lda, &b, &ldb))
            ){
                LAPACKE_chegst(matrix_layout, itype, uplo, n, get_ptr(a), lda, get_cste_ptr(b), ldb);
            }
        } break;
        case zhegst: {
            int matrix_layout; int itype; char uplo; int n; c_binary a; int lda; cste_c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_end}, &matrix_layout, &itype, &uplo, &n, &a, &lda, &b, &ldb))
            ){
                LAPACKE_zhegst(matrix_layout, itype, uplo, n, get_ptr(a), lda, get_cste_ptr(b), ldb);
            }
        } break;
        case chegv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_chegv(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        } break;
        case zhegv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_zhegv(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        } break;
        case chegvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_chegvd(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        } break;
        case zhegvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_zhegvd(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        } break;
        case chegvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &a, &lda, &b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chegvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case zhegvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &a, &lda, &b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhegvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case cherfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cherfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zherfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zherfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cherfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_cherfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case zherfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_zherfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case chesv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_chesv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zhesv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zhesv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case chesvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_chesvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zhesvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zhesvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case chetrd: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &d, &e, &tau))
            ){
                LAPACKE_chetrd(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        } break;
        case zhetrd: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &d, &e, &tau))
            ){
                LAPACKE_zhetrd(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        } break;
        case chetrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_chetrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case zhetrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zhetrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case chetri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_chetri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case zhetri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zhetri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case chetrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_chetrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zhetrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zhetrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case chfrk: {
            int matrix_layout; char transr; char uplo; char trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c))
            ){
                LAPACKE_chfrk(matrix_layout, transr, uplo, trans, n, k, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_double(beta), get_ptr(c));
            }
        } break;
        case zhfrk: {
            int matrix_layout; char transr; char uplo; char trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c))
            ){
                LAPACKE_zhfrk(matrix_layout, transr, uplo, trans, n, k, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_double(beta), get_ptr(c));
            }
        } break;
        case shgeqz: {
            int matrix_layout; char job; char compq; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary t; int ldt; c_binary alphar; c_binary alphai; c_binary beta; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &compq, &compz, &n, &ilo, &ihi, &h, &ldh, &t, &ldt, &alphar, &alphai, &beta, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_shgeqz(matrix_layout, job, compq, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(t), ldt, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz);
            }
        } break;
        case dhgeqz: {
            int matrix_layout; char job; char compq; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary t; int ldt; c_binary alphar; c_binary alphai; c_binary beta; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &compq, &compz, &n, &ilo, &ihi, &h, &ldh, &t, &ldt, &alphar, &alphai, &beta, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_dhgeqz(matrix_layout, job, compq, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(t), ldt, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz);
            }
        } break;
        case chgeqz: {
            int matrix_layout; char job; char compq; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary t; int ldt; c_binary alpha; c_binary beta; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &compq, &compz, &n, &ilo, &ihi, &h, &ldh, &t, &ldt, &alpha, &beta, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_chgeqz(matrix_layout, job, compq, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(t), ldt, get_ptr(alpha), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz);
            }
        } break;
        case zhgeqz: {
            int matrix_layout; char job; char compq; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary t; int ldt; c_binary alpha; c_binary beta; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &compq, &compz, &n, &ilo, &ihi, &h, &ldh, &t, &ldt, &alpha, &beta, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_zhgeqz(matrix_layout, job, compq, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(t), ldt, get_ptr(alpha), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz);
            }
        } break;
        case chpcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_chpcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zhpcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zhpcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case chpev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_chpev(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case zhpev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_zhpev(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case chpevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_chpevd(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case zhpevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_zhpevd(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case chpevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary ap; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ap, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chpevx(matrix_layout, jobz, range, uplo, n, get_ptr(ap), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case zhpevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary ap; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ap, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhpevx(matrix_layout, jobz, range, uplo, n, get_ptr(ap), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case chpgst: {
            int matrix_layout; int itype; char uplo; int n; c_binary ap; cste_c_binary bp;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &itype, &uplo, &n, &ap, &bp))
            ){
                LAPACKE_chpgst(matrix_layout, itype, uplo, n, get_ptr(ap), get_cste_ptr(bp));
            }
        } break;
        case zhpgst: {
            int matrix_layout; int itype; char uplo; int n; c_binary ap; cste_c_binary bp;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &itype, &uplo, &n, &ap, &bp))
            ){
                LAPACKE_zhpgst(matrix_layout, itype, uplo, n, get_ptr(ap), get_cste_ptr(bp));
            }
        } break;
        case chpgv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_chpgv(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case zhpgv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_zhpgv(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case chpgvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_chpgvd(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case zhpgvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_zhpgvd(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case chpgvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary ap; c_binary bp; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &ap, &bp, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chpgvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(ap), get_ptr(bp), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case zhpgvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary ap; c_binary bp; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &ap, &bp, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhpgvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(ap), get_ptr(bp), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case chprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_chprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zhprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zhprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case chpsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_chpsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zhpsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_zhpsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case chpsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_chpsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zhpsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zhpsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case chptrd: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &d, &e, &tau))
            ){
                LAPACKE_chptrd(matrix_layout, uplo, n, get_ptr(ap), get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        } break;
        case zhptrd: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &d, &e, &tau))
            ){
                LAPACKE_zhptrd(matrix_layout, uplo, n, get_ptr(ap), get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        } break;
        case chptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_chptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        } break;
        case zhptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_zhptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        } break;
        case chptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_chptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        } break;
        case zhptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_zhptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        } break;
        case chptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_chptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zhptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_zhptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case shsein: {
            int matrix_layout; char job; char eigsrc; char initv; c_binary select; int n; cste_c_binary h; int ldh; c_binary wr; cste_c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m; c_binary ifaill; c_binary ifailr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &eigsrc, &initv, &select, &n, &h, &ldh, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &mm, &m, &ifaill, &ifailr))
            ){
                LAPACKE_shsein(matrix_layout, job, eigsrc, initv, get_ptr(select), n, get_cste_ptr(h), ldh, get_ptr(wr), get_cste_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m), get_ptr(ifaill), get_ptr(ifailr));
            }
        } break;
        case dhsein: {
            int matrix_layout; char job; char eigsrc; char initv; c_binary select; int n; cste_c_binary h; int ldh; c_binary wr; cste_c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m; c_binary ifaill; c_binary ifailr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &eigsrc, &initv, &select, &n, &h, &ldh, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &mm, &m, &ifaill, &ifailr))
            ){
                LAPACKE_dhsein(matrix_layout, job, eigsrc, initv, get_ptr(select), n, get_cste_ptr(h), ldh, get_ptr(wr), get_cste_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m), get_ptr(ifaill), get_ptr(ifailr));
            }
        } break;
        case chsein: {
            int matrix_layout; char job; char eigsrc; char initv; cste_c_binary select; int n; cste_c_binary h; int ldh; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m; c_binary ifaill; c_binary ifailr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &eigsrc, &initv, &select, &n, &h, &ldh, &w, &vl, &ldvl, &vr, &ldvr, &mm, &m, &ifaill, &ifailr))
            ){
                LAPACKE_chsein(matrix_layout, job, eigsrc, initv, get_cste_ptr(select), n, get_cste_ptr(h), ldh, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m), get_ptr(ifaill), get_ptr(ifailr));
            }
        } break;
        case zhsein: {
            int matrix_layout; char job; char eigsrc; char initv; cste_c_binary select; int n; cste_c_binary h; int ldh; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m; c_binary ifaill; c_binary ifailr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &eigsrc, &initv, &select, &n, &h, &ldh, &w, &vl, &ldvl, &vr, &ldvr, &mm, &m, &ifaill, &ifailr))
            ){
                LAPACKE_zhsein(matrix_layout, job, eigsrc, initv, get_cste_ptr(select), n, get_cste_ptr(h), ldh, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m), get_ptr(ifaill), get_ptr(ifailr));
            }
        } break;
        case shseqr: {
            int matrix_layout; char job; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary wr; c_binary wi; c_binary z; int ldz;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &job, &compz, &n, &ilo, &ihi, &h, &ldh, &wr, &wi, &z, &ldz))
            ){
                LAPACKE_shseqr(matrix_layout, job, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(wr), get_ptr(wi), get_ptr(z), ldz);
            }
        } break;
        case dhseqr: {
            int matrix_layout; char job; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary wr; c_binary wi; c_binary z; int ldz;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &job, &compz, &n, &ilo, &ihi, &h, &ldh, &wr, &wi, &z, &ldz))
            ){
                LAPACKE_dhseqr(matrix_layout, job, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(wr), get_ptr(wi), get_ptr(z), ldz);
            }
        } break;
        case chseqr: {
            int matrix_layout; char job; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &job, &compz, &n, &ilo, &ihi, &h, &ldh, &w, &z, &ldz))
            ){
                LAPACKE_chseqr(matrix_layout, job, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case zhseqr: {
            int matrix_layout; char job; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &job, &compz, &n, &ilo, &ihi, &h, &ldh, &w, &z, &ldz))
            ){
                LAPACKE_zhseqr(matrix_layout, job, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case sopgtr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary tau; c_binary q; int ldq;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &tau, &q, &ldq))
            ){
                LAPACKE_sopgtr(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(q), ldq);
            }
        } break;
        case dopgtr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary tau; c_binary q; int ldq;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &tau, &q, &ldq))
            ){
                LAPACKE_dopgtr(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(q), ldq);
            }
        } break;
        case sopmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary ap; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &ap, &tau, &c, &ldc))
            ){
                LAPACKE_sopmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case dopmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary ap; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &ap, &tau, &c, &ldc))
            ){
                LAPACKE_dopmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case sorgbr: {
            int matrix_layout; char vect; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &vect, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorgbr(matrix_layout, vect, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case dorgbr: {
            int matrix_layout; char vect; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &vect, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorgbr(matrix_layout, vect, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case sorghr: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_sorghr(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case dorghr: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_dorghr(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case sorglq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorglq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case dorglq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorglq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case sorgql: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorgql(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case dorgql: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorgql(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case sorgqr: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorgqr(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case dorgqr: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorgqr(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case sorgrq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorgrq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case dorgrq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorgrq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case sorgtr: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &tau))
            ){
                LAPACKE_sorgtr(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case dorgtr: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &tau))
            ){
                LAPACKE_dorgtr(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case sormbr: {
            int matrix_layout; char vect; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormbr(matrix_layout, vect, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case dormbr: {
            int matrix_layout; char vect; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormbr(matrix_layout, vect, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case sormhr: {
            int matrix_layout; char side; char trans; int m; int n; int ilo; int ihi; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &ilo, &ihi, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormhr(matrix_layout, side, trans, m, n, ilo, ihi, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case dormhr: {
            int matrix_layout; char side; char trans; int m; int n; int ilo; int ihi; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &ilo, &ihi, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormhr(matrix_layout, side, trans, m, n, ilo, ihi, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case sormlq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormlq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case dormlq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormlq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case sormql: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormql(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case dormql: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormql(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case sormqr: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormqr(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case dormqr: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormqr(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case sormrq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormrq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case dormrq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormrq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case sormrz: {
            int matrix_layout; char side; char trans; int m; int n; int k; int l; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &l, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormrz(matrix_layout, side, trans, m, n, k, l, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case dormrz: {
            int matrix_layout; char side; char trans; int m; int n; int k; int l; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &l, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormrz(matrix_layout, side, trans, m, n, k, l, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case sormtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case dormtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case spbcon: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &anorm, &rcond))
            ){
                LAPACKE_spbcon(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case dpbcon: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &anorm, &rcond))
            ){
                LAPACKE_dpbcon(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case cpbcon: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &anorm, &rcond))
            ){
                LAPACKE_cpbcon(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zpbcon: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &anorm, &rcond))
            ){
                LAPACKE_zpbcon(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case spbequ: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &s, &scond, &amax))
            ){
                LAPACKE_spbequ(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case dpbequ: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &s, &scond, &amax))
            ){
                LAPACKE_dpbequ(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case cpbequ: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &s, &scond, &amax))
            ){
                LAPACKE_cpbequ(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case zpbequ: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &s, &scond, &amax))
            ){
                LAPACKE_zpbequ(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case spbrfs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &afb, &ldafb, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_spbrfs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dpbrfs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &afb, &ldafb, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dpbrfs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cpbrfs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &afb, &ldafb, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cpbrfs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zpbrfs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &afb, &ldafb, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zpbrfs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case spbstf: {
            int matrix_layout; char uplo; int n; int kb; c_binary bb; int ldbb;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kb, &bb, &ldbb))
            ){
                LAPACKE_spbstf(matrix_layout, uplo, n, kb, get_ptr(bb), ldbb);
            }
        } break;
        case dpbstf: {
            int matrix_layout; char uplo; int n; int kb; c_binary bb; int ldbb;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kb, &bb, &ldbb))
            ){
                LAPACKE_dpbstf(matrix_layout, uplo, n, kb, get_ptr(bb), ldbb);
            }
        } break;
        case cpbstf: {
            int matrix_layout; char uplo; int n; int kb; c_binary bb; int ldbb;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kb, &bb, &ldbb))
            ){
                LAPACKE_cpbstf(matrix_layout, uplo, n, kb, get_ptr(bb), ldbb);
            }
        } break;
        case zpbstf: {
            int matrix_layout; char uplo; int n; int kb; c_binary bb; int ldbb;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kb, &bb, &ldbb))
            ){
                LAPACKE_zpbstf(matrix_layout, uplo, n, kb, get_ptr(bb), ldbb);
            }
        } break;
        case spbsv: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_spbsv(matrix_layout, uplo, n, kd, nrhs, get_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case dpbsv: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_dpbsv(matrix_layout, uplo, n, kd, nrhs, get_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case cpbsv: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_cpbsv(matrix_layout, uplo, n, kd, nrhs, get_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case zpbsv: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_zpbsv(matrix_layout, uplo, n, kd, nrhs, get_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case spbtrf: {
            int matrix_layout; char uplo; int n; int kd; c_binary ab; int ldab;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab))
            ){
                LAPACKE_spbtrf(matrix_layout, uplo, n, kd, get_ptr(ab), ldab);
            }
        } break;
        case dpbtrf: {
            int matrix_layout; char uplo; int n; int kd; c_binary ab; int ldab;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab))
            ){
                LAPACKE_dpbtrf(matrix_layout, uplo, n, kd, get_ptr(ab), ldab);
            }
        } break;
        case cpbtrf: {
            int matrix_layout; char uplo; int n; int kd; c_binary ab; int ldab;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab))
            ){
                LAPACKE_cpbtrf(matrix_layout, uplo, n, kd, get_ptr(ab), ldab);
            }
        } break;
        case zpbtrf: {
            int matrix_layout; char uplo; int n; int kd; c_binary ab; int ldab;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab))
            ){
                LAPACKE_zpbtrf(matrix_layout, uplo, n, kd, get_ptr(ab), ldab);
            }
        } break;
        case spbtrs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_spbtrs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case dpbtrs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_dpbtrs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case cpbtrs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_cpbtrs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case zpbtrs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_zpbtrs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case spftrf: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_spftrf(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        } break;
        case dpftrf: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_dpftrf(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        } break;
        case cpftrf: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_cpftrf(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        } break;
        case zpftrf: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_zpftrf(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        } break;
        case spftri: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_spftri(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        } break;
        case dpftri: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_dpftri(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        } break;
        case cpftri: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_cpftri(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        } break;
        case zpftri: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_zpftri(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        } break;
        case spftrs: {
            int matrix_layout; char transr; char uplo; int n; int nrhs; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &nrhs, &a, &b, &ldb))
            ){
                LAPACKE_spftrs(matrix_layout, transr, uplo, n, nrhs, get_cste_ptr(a), get_ptr(b), ldb);
            }
        } break;
        case dpftrs: {
            int matrix_layout; char transr; char uplo; int n; int nrhs; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &nrhs, &a, &b, &ldb))
            ){
                LAPACKE_dpftrs(matrix_layout, transr, uplo, n, nrhs, get_cste_ptr(a), get_ptr(b), ldb);
            }
        } break;
        case cpftrs: {
            int matrix_layout; char transr; char uplo; int n; int nrhs; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &nrhs, &a, &b, &ldb))
            ){
                LAPACKE_cpftrs(matrix_layout, transr, uplo, n, nrhs, get_cste_ptr(a), get_ptr(b), ldb);
            }
        } break;
        case zpftrs: {
            int matrix_layout; char transr; char uplo; int n; int nrhs; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &nrhs, &a, &b, &ldb))
            ){
                LAPACKE_zpftrs(matrix_layout, transr, uplo, n, nrhs, get_cste_ptr(a), get_ptr(b), ldb);
            }
        } break;
        case spocon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_spocon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case dpocon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_dpocon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case cpocon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_cpocon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zpocon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_zpocon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case spoequ: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_spoequ(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case dpoequ: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_dpoequ(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case cpoequ: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_cpoequ(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case zpoequ: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_zpoequ(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case spoequb: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_spoequb(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case dpoequb: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_dpoequb(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case cpoequb: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_cpoequb(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case zpoequb: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_zpoequb(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case sporfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sporfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dporfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dporfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cporfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cporfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zporfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zporfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case sporfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_sporfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case dporfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_dporfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case cporfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_cporfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case zporfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_zporfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case sposv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_sposv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case dposv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dposv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case cposv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_cposv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case zposv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_zposv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case spotrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_spotrf(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        } break;
        case dpotrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_dpotrf(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        } break;
        case cpotrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_cpotrf(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        } break;
        case zpotrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_zpotrf(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        } break;
        case spotri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_spotri(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        } break;
        case dpotri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_dpotri(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        } break;
        case cpotri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_cpotri(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        } break;
        case zpotri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_zpotri(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        } break;
        case spotrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_spotrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case dpotrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dpotrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case cpotrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_cpotrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case zpotrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_zpotrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case sppcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &anorm, &rcond))
            ){
                LAPACKE_sppcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case dppcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &anorm, &rcond))
            ){
                LAPACKE_dppcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case cppcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &anorm, &rcond))
            ){
                LAPACKE_cppcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zppcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &anorm, &rcond))
            ){
                LAPACKE_zppcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case sppequ: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &s, &scond, &amax))
            ){
                LAPACKE_sppequ(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case dppequ: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &s, &scond, &amax))
            ){
                LAPACKE_dppequ(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case cppequ: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &s, &scond, &amax))
            ){
                LAPACKE_cppequ(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case zppequ: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &s, &scond, &amax))
            ){
                LAPACKE_zppequ(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case spprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_spprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dpprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dpprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cpprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cpprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zpprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zpprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case sppsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_sppsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case dppsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_dppsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case cppsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_cppsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case zppsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_zppsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case spptrf: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_spptrf(matrix_layout, uplo, n, get_ptr(ap));
            }
        } break;
        case dpptrf: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_dpptrf(matrix_layout, uplo, n, get_ptr(ap));
            }
        } break;
        case cpptrf: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_cpptrf(matrix_layout, uplo, n, get_ptr(ap));
            }
        } break;
        case zpptrf: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_zpptrf(matrix_layout, uplo, n, get_ptr(ap));
            }
        } break;
        case spptri: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_spptri(matrix_layout, uplo, n, get_ptr(ap));
            }
        } break;
        case dpptri: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_dpptri(matrix_layout, uplo, n, get_ptr(ap));
            }
        } break;
        case cpptri: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_cpptri(matrix_layout, uplo, n, get_ptr(ap));
            }
        } break;
        case zpptri: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_zpptri(matrix_layout, uplo, n, get_ptr(ap));
            }
        } break;
        case spptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_spptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case dpptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_dpptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case cpptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_cpptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case zpptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_zpptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case spstrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary piv; c_binary rank; cste_c_binary tol;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &piv, &rank, &tol))
            ){
                LAPACKE_spstrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(piv), get_ptr(rank), get_cste_double(tol));
            }
        } break;
        case dpstrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary piv; c_binary rank; cste_c_binary tol;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &piv, &rank, &tol))
            ){
                LAPACKE_dpstrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(piv), get_ptr(rank), get_cste_double(tol));
            }
        } break;
        case cpstrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary piv; c_binary rank; cste_c_binary tol;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &piv, &rank, &tol))
            ){
                LAPACKE_cpstrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(piv), get_ptr(rank), get_cste_double(tol));
            }
        } break;
        case zpstrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary piv; c_binary rank; cste_c_binary tol;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &piv, &rank, &tol))
            ){
                LAPACKE_zpstrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(piv), get_ptr(rank), get_cste_double(tol));
            }
        } break;
        case sptcon: {
            int n; cste_c_binary d; cste_c_binary e; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &n, &d, &e, &anorm, &rcond))
            ){
                LAPACKE_sptcon(n, get_cste_ptr(d), get_cste_ptr(e), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case dptcon: {
            int n; cste_c_binary d; cste_c_binary e; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &n, &d, &e, &anorm, &rcond))
            ){
                LAPACKE_dptcon(n, get_cste_ptr(d), get_cste_ptr(e), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case cptcon: {
            int n; cste_c_binary d; cste_c_binary e; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &n, &d, &e, &anorm, &rcond))
            ){
                LAPACKE_cptcon(n, get_cste_ptr(d), get_cste_ptr(e), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zptcon: {
            int n; cste_c_binary d; cste_c_binary e; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &n, &d, &e, &anorm, &rcond))
            ){
                LAPACKE_zptcon(n, get_cste_ptr(d), get_cste_ptr(e), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case spteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_spteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case dpteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dpteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case cpteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_cpteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case zpteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_zpteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case sptrfs: {
            int matrix_layout; int n; int nrhs; cste_c_binary d; cste_c_binary e; cste_c_binary df; cste_c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sptrfs(matrix_layout, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_cste_ptr(df), get_cste_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dptrfs: {
            int matrix_layout; int n; int nrhs; cste_c_binary d; cste_c_binary e; cste_c_binary df; cste_c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dptrfs(matrix_layout, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_cste_ptr(df), get_cste_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cptrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary d; cste_c_binary e; cste_c_binary df; cste_c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cptrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_cste_ptr(df), get_cste_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zptrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary d; cste_c_binary e; cste_c_binary df; cste_c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zptrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_cste_ptr(df), get_cste_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case sptsv: {
            int matrix_layout; int n; int nrhs; c_binary d; c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_sptsv(matrix_layout, n, nrhs, get_ptr(d), get_ptr(e), get_ptr(b), ldb);
            }
        } break;
        case dptsv: {
            int matrix_layout; int n; int nrhs; c_binary d; c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_dptsv(matrix_layout, n, nrhs, get_ptr(d), get_ptr(e), get_ptr(b), ldb);
            }
        } break;
        case cptsv: {
            int matrix_layout; int n; int nrhs; c_binary d; c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_cptsv(matrix_layout, n, nrhs, get_ptr(d), get_ptr(e), get_ptr(b), ldb);
            }
        } break;
        case zptsv: {
            int matrix_layout; int n; int nrhs; c_binary d; c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_zptsv(matrix_layout, n, nrhs, get_ptr(d), get_ptr(e), get_ptr(b), ldb);
            }
        } break;
        case sptsvx: {
            int matrix_layout; char fact; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary df; c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_sptsvx(matrix_layout, fact, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(df), get_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dptsvx: {
            int matrix_layout; char fact; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary df; c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_dptsvx(matrix_layout, fact, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(df), get_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cptsvx: {
            int matrix_layout; char fact; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary df; c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_cptsvx(matrix_layout, fact, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(df), get_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zptsvx: {
            int matrix_layout; char fact; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary df; c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zptsvx(matrix_layout, fact, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(df), get_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case spttrf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_spttrf(n, get_ptr(d), get_ptr(e));
            }
        } break;
        case dpttrf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_dpttrf(n, get_ptr(d), get_ptr(e));
            }
        } break;
        case cpttrf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_cpttrf(n, get_ptr(d), get_ptr(e));
            }
        } break;
        case zpttrf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_zpttrf(n, get_ptr(d), get_ptr(e));
            }
        } break;
        case spttrs: {
            int matrix_layout; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_spttrs(matrix_layout, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(b), ldb);
            }
        } break;
        case dpttrs: {
            int matrix_layout; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_dpttrs(matrix_layout, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(b), ldb);
            }
        } break;
        case cpttrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_cpttrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(b), ldb);
            }
        } break;
        case zpttrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_zpttrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(b), ldb);
            }
        } break;
        case ssbev: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_ssbev(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case dsbev: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_dsbev(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case ssbevd: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_ssbevd(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case dsbevd: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_dsbevd(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case ssbevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int kd; c_binary ab; int ldab; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &kd, &ab, &ldab, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_ssbevx(matrix_layout, jobz, range, uplo, n, kd, get_ptr(ab), ldab, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case dsbevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int kd; c_binary ab; int ldab; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &kd, &ab, &ldab, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dsbevx(matrix_layout, jobz, range, uplo, n, kd, get_ptr(ab), ldab, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case ssbgst: {
            int matrix_layout; char vect; char uplo; int n; int ka; int kb; c_binary ab; int ldab; cste_c_binary bb; int ldbb; c_binary x; int ldx;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &x, &ldx))
            ){
                LAPACKE_ssbgst(matrix_layout, vect, uplo, n, ka, kb, get_ptr(ab), ldab, get_cste_ptr(bb), ldbb, get_ptr(x), ldx);
            }
        } break;
        case dsbgst: {
            int matrix_layout; char vect; char uplo; int n; int ka; int kb; c_binary ab; int ldab; cste_c_binary bb; int ldbb; c_binary x; int ldx;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &x, &ldx))
            ){
                LAPACKE_dsbgst(matrix_layout, vect, uplo, n, ka, kb, get_ptr(ab), ldab, get_cste_ptr(bb), ldbb, get_ptr(x), ldx);
            }
        } break;
        case ssbgv: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_ssbgv(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case dsbgv: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_dsbgv(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case ssbgvd: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_ssbgvd(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case dsbgvd: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_dsbgvd(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case ssbgvx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_ssbgvx(matrix_layout, jobz, range, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case dsbgvx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dsbgvx(matrix_layout, jobz, range, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case ssbtrd: {
            int matrix_layout; char vect; char uplo; int n; int kd; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &kd, &ab, &ldab, &d, &e, &q, &ldq))
            ){
                LAPACKE_ssbtrd(matrix_layout, vect, uplo, n, kd, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq);
            }
        } break;
        case dsbtrd: {
            int matrix_layout; char vect; char uplo; int n; int kd; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &kd, &ab, &ldab, &d, &e, &q, &ldq))
            ){
                LAPACKE_dsbtrd(matrix_layout, vect, uplo, n, kd, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq);
            }
        } break;
        case ssfrk: {
            int matrix_layout; char transr; char uplo; char trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c))
            ){
                LAPACKE_ssfrk(matrix_layout, transr, uplo, trans, n, k, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_double(beta), get_ptr(c));
            }
        } break;
        case dsfrk: {
            int matrix_layout; char transr; char uplo; char trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c))
            ){
                LAPACKE_dsfrk(matrix_layout, transr, uplo, trans, n, k, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_double(beta), get_ptr(c));
            }
        } break;
        case sspcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_sspcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case dspcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_dspcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case cspcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_cspcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zspcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zspcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case sspev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_sspev(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case dspev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_dspev(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case sspevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_sspevd(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case dspevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_dspevd(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case sspevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary ap; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ap, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_sspevx(matrix_layout, jobz, range, uplo, n, get_ptr(ap), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case dspevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary ap; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ap, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dspevx(matrix_layout, jobz, range, uplo, n, get_ptr(ap), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case sspgst: {
            int matrix_layout; int itype; char uplo; int n; c_binary ap; cste_c_binary bp;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &itype, &uplo, &n, &ap, &bp))
            ){
                LAPACKE_sspgst(matrix_layout, itype, uplo, n, get_ptr(ap), get_cste_ptr(bp));
            }
        } break;
        case dspgst: {
            int matrix_layout; int itype; char uplo; int n; c_binary ap; cste_c_binary bp;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &itype, &uplo, &n, &ap, &bp))
            ){
                LAPACKE_dspgst(matrix_layout, itype, uplo, n, get_ptr(ap), get_cste_ptr(bp));
            }
        } break;
        case sspgv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_sspgv(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case dspgv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_dspgv(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case sspgvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_sspgvd(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case dspgvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_dspgvd(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        } break;
        case sspgvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary ap; c_binary bp; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &ap, &bp, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_sspgvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(ap), get_ptr(bp), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case dspgvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary ap; c_binary bp; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &ap, &bp, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dspgvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(ap), get_ptr(bp), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case ssprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ssprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dsprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dsprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case csprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_csprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zsprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zsprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case sspsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_sspsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case dspsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_dspsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case cspsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_cspsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zspsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_zspsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case sspsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_sspsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dspsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_dspsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case cspsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_cspsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zspsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zspsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case ssptrd: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &d, &e, &tau))
            ){
                LAPACKE_ssptrd(matrix_layout, uplo, n, get_ptr(ap), get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        } break;
        case dsptrd: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &d, &e, &tau))
            ){
                LAPACKE_dsptrd(matrix_layout, uplo, n, get_ptr(ap), get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        } break;
        case ssptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_ssptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        } break;
        case dsptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_dsptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        } break;
        case csptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_csptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        } break;
        case zsptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_zsptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        } break;
        case ssptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_ssptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        } break;
        case dsptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_dsptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        } break;
        case csptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_csptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        } break;
        case zsptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_zsptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        } break;
        case ssptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_ssptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case dsptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_dsptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case csptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_csptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zsptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_zsptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case sstebz: {
            char range; char order; int n; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; cste_c_binary d; cste_c_binary e; c_binary m; c_binary nsplit; c_binary w; c_binary iblock; c_binary isplit;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_char, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &range, &order, &n, &vl, &vu, &il, &iu, &abstol, &d, &e, &m, &nsplit, &w, &iblock, &isplit))
            ){
                LAPACKE_sstebz(range, order, n, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_cste_ptr(d), get_cste_ptr(e), get_ptr(m), get_ptr(nsplit), get_ptr(w), get_ptr(iblock), get_ptr(isplit));
            }
        } break;
        case dstebz: {
            char range; char order; int n; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; cste_c_binary d; cste_c_binary e; c_binary m; c_binary nsplit; c_binary w; c_binary iblock; c_binary isplit;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_char, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &range, &order, &n, &vl, &vu, &il, &iu, &abstol, &d, &e, &m, &nsplit, &w, &iblock, &isplit))
            ){
                LAPACKE_dstebz(range, order, n, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_cste_ptr(d), get_cste_ptr(e), get_ptr(m), get_ptr(nsplit), get_ptr(w), get_ptr(iblock), get_ptr(isplit));
            }
        } break;
        case sstedc: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_sstedc(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case dstedc: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dstedc(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case cstedc: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_cstedc(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case zstedc: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_zstedc(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case sstegr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_sstegr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case dstegr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_dstegr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case cstegr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_cstegr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case zstegr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_zstegr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case sstein: {
            int matrix_layout; int n; cste_c_binary d; cste_c_binary e; int m; cste_c_binary w; cste_c_binary iblock; cste_c_binary isplit; c_binary z; int ldz; c_binary ifailv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &d, &e, &m, &w, &iblock, &isplit, &z, &ldz, &ifailv))
            ){
                LAPACKE_sstein(matrix_layout, n, get_cste_ptr(d), get_cste_ptr(e), m, get_cste_ptr(w), get_cste_ptr(iblock), get_cste_ptr(isplit), get_ptr(z), ldz, get_ptr(ifailv));
            }
        } break;
        case dstein: {
            int matrix_layout; int n; cste_c_binary d; cste_c_binary e; int m; cste_c_binary w; cste_c_binary iblock; cste_c_binary isplit; c_binary z; int ldz; c_binary ifailv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &d, &e, &m, &w, &iblock, &isplit, &z, &ldz, &ifailv))
            ){
                LAPACKE_dstein(matrix_layout, n, get_cste_ptr(d), get_cste_ptr(e), m, get_cste_ptr(w), get_cste_ptr(iblock), get_cste_ptr(isplit), get_ptr(z), ldz, get_ptr(ifailv));
            }
        } break;
        case cstein: {
            int matrix_layout; int n; cste_c_binary d; cste_c_binary e; int m; cste_c_binary w; cste_c_binary iblock; cste_c_binary isplit; c_binary z; int ldz; c_binary ifailv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &d, &e, &m, &w, &iblock, &isplit, &z, &ldz, &ifailv))
            ){
                LAPACKE_cstein(matrix_layout, n, get_cste_ptr(d), get_cste_ptr(e), m, get_cste_ptr(w), get_cste_ptr(iblock), get_cste_ptr(isplit), get_ptr(z), ldz, get_ptr(ifailv));
            }
        } break;
        case zstein: {
            int matrix_layout; int n; cste_c_binary d; cste_c_binary e; int m; cste_c_binary w; cste_c_binary iblock; cste_c_binary isplit; c_binary z; int ldz; c_binary ifailv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &d, &e, &m, &w, &iblock, &isplit, &z, &ldz, &ifailv))
            ){
                LAPACKE_zstein(matrix_layout, n, get_cste_ptr(d), get_cste_ptr(e), m, get_cste_ptr(w), get_cste_ptr(iblock), get_cste_ptr(isplit), get_ptr(z), ldz, get_ptr(ifailv));
            }
        } break;
        case sstemr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; c_binary m; c_binary w; c_binary z; int ldz; int nzc; c_binary isuppz; c_binary tryrac;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &m, &w, &z, &ldz, &nzc, &isuppz, &tryrac))
            ){
                LAPACKE_sstemr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_ptr(m), get_ptr(w), get_ptr(z), ldz, nzc, get_ptr(isuppz), get_ptr(tryrac));
            }
        } break;
        case dstemr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; c_binary m; c_binary w; c_binary z; int ldz; int nzc; c_binary isuppz; c_binary tryrac;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &m, &w, &z, &ldz, &nzc, &isuppz, &tryrac))
            ){
                LAPACKE_dstemr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_ptr(m), get_ptr(w), get_ptr(z), ldz, nzc, get_ptr(isuppz), get_ptr(tryrac));
            }
        } break;
        case cstemr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; c_binary m; c_binary w; c_binary z; int ldz; int nzc; c_binary isuppz; c_binary tryrac;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &m, &w, &z, &ldz, &nzc, &isuppz, &tryrac))
            ){
                LAPACKE_cstemr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_ptr(m), get_ptr(w), get_ptr(z), ldz, nzc, get_ptr(isuppz), get_ptr(tryrac));
            }
        } break;
        case zstemr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; c_binary m; c_binary w; c_binary z; int ldz; int nzc; c_binary isuppz; c_binary tryrac;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &m, &w, &z, &ldz, &nzc, &isuppz, &tryrac))
            ){
                LAPACKE_zstemr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_ptr(m), get_ptr(w), get_ptr(z), ldz, nzc, get_ptr(isuppz), get_ptr(tryrac));
            }
        } break;
        case ssteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_ssteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case dsteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dsteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case csteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_csteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case zsteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_zsteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case ssterf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_ssterf(n, get_ptr(d), get_ptr(e));
            }
        } break;
        case dsterf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_dsterf(n, get_ptr(d), get_ptr(e));
            }
        } break;
        case sstev: {
            int matrix_layout; char jobz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_sstev(matrix_layout, jobz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case dstev: {
            int matrix_layout; char jobz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dstev(matrix_layout, jobz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case sstevd: {
            int matrix_layout; char jobz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_sstevd(matrix_layout, jobz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case dstevd: {
            int matrix_layout; char jobz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dstevd(matrix_layout, jobz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        } break;
        case sstevr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_sstevr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case dstevr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_dstevr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case sstevx: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_sstevx(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case dstevx: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dstevx(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case ssycon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_ssycon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case dsycon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_dsycon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case csycon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_csycon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case zsycon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zsycon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        } break;
        case ssyequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_ssyequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case dsyequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_dsyequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case csyequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_csyequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case zsyequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_zsyequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        } break;
        case ssyev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_ssyev(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        } break;
        case dsyev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_dsyev(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        } break;
        case ssyevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_ssyevd(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        } break;
        case dsyevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        } break;
        case ssyevr: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_ssyevr(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case dsyevr: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_dsyevr(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        } break;
        case ssyevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_ssyevx(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case dsyevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dsyevx(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case ssygst: {
            int matrix_layout; int itype; char uplo; int n; c_binary a; int lda; cste_c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_end}, &matrix_layout, &itype, &uplo, &n, &a, &lda, &b, &ldb))
            ){
                LAPACKE_ssygst(matrix_layout, itype, uplo, n, get_ptr(a), lda, get_cste_ptr(b), ldb);
            }
        } break;
        case dsygst: {
            int matrix_layout; int itype; char uplo; int n; c_binary a; int lda; cste_c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_end}, &matrix_layout, &itype, &uplo, &n, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dsygst(matrix_layout, itype, uplo, n, get_ptr(a), lda, get_cste_ptr(b), ldb);
            }
        } break;
        case ssygv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_ssygv(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        } break;
        case dsygv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_dsygv(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        } break;
        case ssygvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_ssygvd(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        } break;
        case dsygvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_dsygvd(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        } break;
        case ssygvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &a, &lda, &b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_ssygvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case dsygvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &a, &lda, &b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dsygvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        } break;
        case ssyrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ssyrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dsyrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dsyrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case csyrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_csyrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zsyrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zsyrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case ssyrfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_ssyrfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case dsyrfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_dsyrfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case csyrfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_csyrfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case zsyrfsx: {
            int matrix_layout; char uplo; char equed; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary s; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary berr; int n_err_bnds; c_binary err_bnds_norm; c_binary err_bnds_comp; int nparams; c_binary params;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &equed, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &s, &b, &ldb, &x, &ldx, &rcond, &berr, &n_err_bnds, &err_bnds_norm, &err_bnds_comp, &nparams, &params))
            ){
                LAPACKE_zsyrfsx(matrix_layout, uplo, equed, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(s), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(berr), n_err_bnds, get_ptr(err_bnds_norm), get_ptr(err_bnds_comp), nparams, get_ptr(params));
            }
        } break;
        case ssysv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_ssysv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case dsysv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_dsysv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case csysv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_csysv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zsysv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zsysv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case ssysvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_ssysvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dsysvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_dsysvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case csysvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_csysvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case zsysvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zsysvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case ssytrd: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &d, &e, &tau))
            ){
                LAPACKE_ssytrd(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        } break;
        case dsytrd: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &d, &e, &tau))
            ){
                LAPACKE_dsytrd(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        } break;
        case ssytrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_ssytrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case dsytrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_dsytrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case csytrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_csytrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case zsytrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zsytrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        } break;
        case ssytri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_ssytri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case dsytri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_dsytri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case csytri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_csytri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case zsytri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zsytri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        } break;
        case ssytrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_ssytrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case dsytrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_dsytrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case csytrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_csytrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case zsytrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zsytrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        } break;
        case stbcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; int kd; cste_c_binary ab; int ldab; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &kd, &ab, &ldab, &rcond))
            ){
                LAPACKE_stbcon(matrix_layout, norm, uplo, diag, n, kd, get_cste_ptr(ab), ldab, get_ptr(rcond));
            }
        } break;
        case dtbcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; int kd; cste_c_binary ab; int ldab; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &kd, &ab, &ldab, &rcond))
            ){
                LAPACKE_dtbcon(matrix_layout, norm, uplo, diag, n, kd, get_cste_ptr(ab), ldab, get_ptr(rcond));
            }
        } break;
        case ctbcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; int kd; cste_c_binary ab; int ldab; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &kd, &ab, &ldab, &rcond))
            ){
                LAPACKE_ctbcon(matrix_layout, norm, uplo, diag, n, kd, get_cste_ptr(ab), ldab, get_ptr(rcond));
            }
        } break;
        case ztbcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; int kd; cste_c_binary ab; int ldab; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &kd, &ab, &ldab, &rcond))
            ){
                LAPACKE_ztbcon(matrix_layout, norm, uplo, diag, n, kd, get_cste_ptr(ab), ldab, get_ptr(rcond));
            }
        } break;
        case stbrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_stbrfs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dtbrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dtbrfs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case ctbrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ctbrfs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case ztbrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ztbrfs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case stbtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_stbtrs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case dtbtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_dtbtrs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case ctbtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_ctbtrs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case ztbtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_ztbtrs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        } break;
        case stfsm: {
            int matrix_layout; char transr; char side; char uplo; char trans; char diag; int m; int n; cste_c_binary alpha; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &side, &uplo, &trans, &diag, &m, &n, &alpha, &a, &b, &ldb))
            ){
                LAPACKE_stfsm(matrix_layout, transr, side, uplo, trans, diag, m, n, get_cste_double(alpha), get_cste_ptr(a), get_ptr(b), ldb);
            }
        } break;
        case dtfsm: {
            int matrix_layout; char transr; char side; char uplo; char trans; char diag; int m; int n; cste_c_binary alpha; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &side, &uplo, &trans, &diag, &m, &n, &alpha, &a, &b, &ldb))
            ){
                LAPACKE_dtfsm(matrix_layout, transr, side, uplo, trans, diag, m, n, get_cste_double(alpha), get_cste_ptr(a), get_ptr(b), ldb);
            }
        } break;
        case stftri: {
            int matrix_layout; char transr; char uplo; char diag; int n; c_binary a;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &diag, &n, &a))
            ){
                LAPACKE_stftri(matrix_layout, transr, uplo, diag, n, get_ptr(a));
            }
        } break;
        case dtftri: {
            int matrix_layout; char transr; char uplo; char diag; int n; c_binary a;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &diag, &n, &a))
            ){
                LAPACKE_dtftri(matrix_layout, transr, uplo, diag, n, get_ptr(a));
            }
        } break;
        case ctftri: {
            int matrix_layout; char transr; char uplo; char diag; int n; c_binary a;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &diag, &n, &a))
            ){
                LAPACKE_ctftri(matrix_layout, transr, uplo, diag, n, get_ptr(a));
            }
        } break;
        case ztftri: {
            int matrix_layout; char transr; char uplo; char diag; int n; c_binary a;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &diag, &n, &a))
            ){
                LAPACKE_ztftri(matrix_layout, transr, uplo, diag, n, get_ptr(a));
            }
        } break;
        case stfttp: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &ap))
            ){
                LAPACKE_stfttp(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(ap));
            }
        } break;
        case dtfttp: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &ap))
            ){
                LAPACKE_dtfttp(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(ap));
            }
        } break;
        case ctfttp: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &ap))
            ){
                LAPACKE_ctfttp(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(ap));
            }
        } break;
        case ztfttp: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &ap))
            ){
                LAPACKE_ztfttp(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(ap));
            }
        } break;
        case stfttr: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary a; int lda;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &a, &lda))
            ){
                LAPACKE_stfttr(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(a), lda);
            }
        } break;
        case dtfttr: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary a; int lda;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &a, &lda))
            ){
                LAPACKE_dtfttr(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(a), lda);
            }
        } break;
        case ctfttr: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary a; int lda;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &a, &lda))
            ){
                LAPACKE_ctfttr(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(a), lda);
            }
        } break;
        case ztfttr: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary a; int lda;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &a, &lda))
            ){
                LAPACKE_ztfttr(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(a), lda);
            }
        } break;
        case stgevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; cste_c_binary s; int lds; cste_c_binary p; int ldp; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &s, &lds, &p, &ldp, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_stgevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_cste_ptr(s), lds, get_cste_ptr(p), ldp, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        } break;
        case dtgevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; cste_c_binary s; int lds; cste_c_binary p; int ldp; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &s, &lds, &p, &ldp, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_dtgevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_cste_ptr(s), lds, get_cste_ptr(p), ldp, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        } break;
        case ctgevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; cste_c_binary s; int lds; cste_c_binary p; int ldp; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &s, &lds, &p, &ldp, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_ctgevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_cste_ptr(s), lds, get_cste_ptr(p), ldp, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        } break;
        case ztgevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; cste_c_binary s; int lds; cste_c_binary p; int ldp; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &s, &lds, &p, &ldp, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_ztgevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_cste_ptr(s), lds, get_cste_ptr(p), ldp, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        } break;
        case stgexc: {
            int matrix_layout; int wantq; int wantz; int n; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz; c_binary ifst; c_binary ilst;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &wantq, &wantz, &n, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz, &ifst, &ilst))
            ){
                LAPACKE_stgexc(matrix_layout, wantq, wantz, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(ifst), get_ptr(ilst));
            }
        } break;
        case dtgexc: {
            int matrix_layout; int wantq; int wantz; int n; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz; c_binary ifst; c_binary ilst;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &wantq, &wantz, &n, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz, &ifst, &ilst))
            ){
                LAPACKE_dtgexc(matrix_layout, wantq, wantz, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(ifst), get_ptr(ilst));
            }
        } break;
        case ctgexc: {
            int matrix_layout; int wantq; int wantz; int n; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz; int ifst; int ilst;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_int, e_end}, &matrix_layout, &wantq, &wantz, &n, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz, &ifst, &ilst))
            ){
                LAPACKE_ctgexc(matrix_layout, wantq, wantz, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz, ifst, ilst);
            }
        } break;
        case ztgexc: {
            int matrix_layout; int wantq; int wantz; int n; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz; int ifst; int ilst;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_int, e_end}, &matrix_layout, &wantq, &wantz, &n, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz, &ifst, &ilst))
            ){
                LAPACKE_ztgexc(matrix_layout, wantq, wantz, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz, ifst, ilst);
            }
        } break;
        case stgsen: {
            int matrix_layout; int ijob; int wantq; int wantz; cste_c_binary select; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary q; int ldq; c_binary z; int ldz; c_binary m; c_binary pl; c_binary pr; c_binary dif;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &ijob, &wantq, &wantz, &select, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &q, &ldq, &z, &ldz, &m, &pl, &pr, &dif))
            ){
                LAPACKE_stgsen(matrix_layout, ijob, wantq, wantz, get_cste_ptr(select), n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(m), get_ptr(pl), get_ptr(pr), get_ptr(dif));
            }
        } break;
        case dtgsen: {
            int matrix_layout; int ijob; int wantq; int wantz; cste_c_binary select; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary q; int ldq; c_binary z; int ldz; c_binary m; c_binary pl; c_binary pr; c_binary dif;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &ijob, &wantq, &wantz, &select, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &q, &ldq, &z, &ldz, &m, &pl, &pr, &dif))
            ){
                LAPACKE_dtgsen(matrix_layout, ijob, wantq, wantz, get_cste_ptr(select), n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(m), get_ptr(pl), get_ptr(pr), get_ptr(dif));
            }
        } break;
        case ctgsen: {
            int matrix_layout; int ijob; int wantq; int wantz; cste_c_binary select; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary q; int ldq; c_binary z; int ldz; c_binary m; c_binary pl; c_binary pr; c_binary dif;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &ijob, &wantq, &wantz, &select, &n, &a, &lda, &b, &ldb, &alpha, &beta, &q, &ldq, &z, &ldz, &m, &pl, &pr, &dif))
            ){
                LAPACKE_ctgsen(matrix_layout, ijob, wantq, wantz, get_cste_ptr(select), n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(m), get_ptr(pl), get_ptr(pr), get_ptr(dif));
            }
        } break;
        case ztgsen: {
            int matrix_layout; int ijob; int wantq; int wantz; cste_c_binary select; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary q; int ldq; c_binary z; int ldz; c_binary m; c_binary pl; c_binary pr; c_binary dif;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &ijob, &wantq, &wantz, &select, &n, &a, &lda, &b, &ldb, &alpha, &beta, &q, &ldq, &z, &ldz, &m, &pl, &pr, &dif))
            ){
                LAPACKE_ztgsen(matrix_layout, ijob, wantq, wantz, get_cste_ptr(select), n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(m), get_ptr(pl), get_ptr(pr), get_ptr(dif));
            }
        } break;
        case stgsja: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; int k; int l; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary ncycle;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, &a, &lda, &b, &ldb, &tola, &tolb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &ncycle))
            ){
                LAPACKE_stgsja(matrix_layout, jobu, jobv, jobq, m, p, n, k, l, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(ncycle));
            }
        } break;
        case dtgsja: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; int k; int l; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary ncycle;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, &a, &lda, &b, &ldb, &tola, &tolb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &ncycle))
            ){
                LAPACKE_dtgsja(matrix_layout, jobu, jobv, jobq, m, p, n, k, l, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(ncycle));
            }
        } break;
        case ctgsja: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; int k; int l; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary ncycle;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, &a, &lda, &b, &ldb, &tola, &tolb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &ncycle))
            ){
                LAPACKE_ctgsja(matrix_layout, jobu, jobv, jobq, m, p, n, k, l, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(ncycle));
            }
        } break;
        case ztgsja: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; int k; int l; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary ncycle;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, &a, &lda, &b, &ldb, &tola, &tolb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &ncycle))
            ){
                LAPACKE_ztgsja(matrix_layout, jobu, jobv, jobq, m, p, n, k, l, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(ncycle));
            }
        } break;
        case stgsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary dif; int mm; c_binary m;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &a, &lda, &b, &ldb, &vl, &ldvl, &vr, &ldvr, &s, &dif, &mm, &m))
            ){
                LAPACKE_stgsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(dif), mm, get_ptr(m));
            }
        } break;
        case dtgsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary dif; int mm; c_binary m;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &a, &lda, &b, &ldb, &vl, &ldvl, &vr, &ldvr, &s, &dif, &mm, &m))
            ){
                LAPACKE_dtgsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(dif), mm, get_ptr(m));
            }
        } break;
        case ctgsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary dif; int mm; c_binary m;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &a, &lda, &b, &ldb, &vl, &ldvl, &vr, &ldvr, &s, &dif, &mm, &m))
            ){
                LAPACKE_ctgsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(dif), mm, get_ptr(m));
            }
        } break;
        case ztgsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary dif; int mm; c_binary m;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &a, &lda, &b, &ldb, &vl, &ldvl, &vr, &ldvr, &s, &dif, &mm, &m))
            ){
                LAPACKE_ztgsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(dif), mm, get_ptr(m));
            }
        } break;
        case stgsyl: {
            int matrix_layout; char trans; int ijob; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; cste_c_binary d; int ldd; cste_c_binary e; int lde; c_binary f; int ldf; c_binary scale; c_binary dif;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &ijob, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &d, &ldd, &e, &lde, &f, &ldf, &scale, &dif))
            ){
                LAPACKE_stgsyl(matrix_layout, trans, ijob, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_cste_ptr(d), ldd, get_cste_ptr(e), lde, get_ptr(f), ldf, get_ptr(scale), get_ptr(dif));
            }
        } break;
        case dtgsyl: {
            int matrix_layout; char trans; int ijob; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; cste_c_binary d; int ldd; cste_c_binary e; int lde; c_binary f; int ldf; c_binary scale; c_binary dif;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &ijob, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &d, &ldd, &e, &lde, &f, &ldf, &scale, &dif))
            ){
                LAPACKE_dtgsyl(matrix_layout, trans, ijob, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_cste_ptr(d), ldd, get_cste_ptr(e), lde, get_ptr(f), ldf, get_ptr(scale), get_ptr(dif));
            }
        } break;
        case ctgsyl: {
            int matrix_layout; char trans; int ijob; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; cste_c_binary d; int ldd; cste_c_binary e; int lde; c_binary f; int ldf; c_binary scale; c_binary dif;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &ijob, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &d, &ldd, &e, &lde, &f, &ldf, &scale, &dif))
            ){
                LAPACKE_ctgsyl(matrix_layout, trans, ijob, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_cste_ptr(d), ldd, get_cste_ptr(e), lde, get_ptr(f), ldf, get_ptr(scale), get_ptr(dif));
            }
        } break;
        case ztgsyl: {
            int matrix_layout; char trans; int ijob; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; cste_c_binary d; int ldd; cste_c_binary e; int lde; c_binary f; int ldf; c_binary scale; c_binary dif;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &ijob, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &d, &ldd, &e, &lde, &f, &ldf, &scale, &dif))
            ){
                LAPACKE_ztgsyl(matrix_layout, trans, ijob, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_cste_ptr(d), ldd, get_cste_ptr(e), lde, get_ptr(f), ldf, get_ptr(scale), get_ptr(dif));
            }
        } break;
        case stpcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary ap; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &ap, &rcond))
            ){
                LAPACKE_stpcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(ap), get_ptr(rcond));
            }
        } break;
        case dtpcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary ap; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &ap, &rcond))
            ){
                LAPACKE_dtpcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(ap), get_ptr(rcond));
            }
        } break;
        case ctpcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary ap; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &ap, &rcond))
            ){
                LAPACKE_ctpcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(ap), get_ptr(rcond));
            }
        } break;
        case ztpcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary ap; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &ap, &rcond))
            ){
                LAPACKE_ztpcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(ap), get_ptr(rcond));
            }
        } break;
        case stprfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_stprfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dtprfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dtprfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case ctprfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ctprfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case ztprfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ztprfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case stptri: {
            int matrix_layout; char uplo; char diag; int n; c_binary ap;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &diag, &n, &ap))
            ){
                LAPACKE_stptri(matrix_layout, uplo, diag, n, get_ptr(ap));
            }
        } break;
        case dtptri: {
            int matrix_layout; char uplo; char diag; int n; c_binary ap;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &diag, &n, &ap))
            ){
                LAPACKE_dtptri(matrix_layout, uplo, diag, n, get_ptr(ap));
            }
        } break;
        case ctptri: {
            int matrix_layout; char uplo; char diag; int n; c_binary ap;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &diag, &n, &ap))
            ){
                LAPACKE_ctptri(matrix_layout, uplo, diag, n, get_ptr(ap));
            }
        } break;
        case ztptri: {
            int matrix_layout; char uplo; char diag; int n; c_binary ap;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &diag, &n, &ap))
            ){
                LAPACKE_ztptri(matrix_layout, uplo, diag, n, get_ptr(ap));
            }
        } break;
        case stptrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_stptrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case dtptrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_dtptrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case ctptrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_ctptrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case ztptrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_ztptrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        } break;
        case stpttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary ap; c_binary arf;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &ap, &arf))
            ){
                LAPACKE_stpttf(matrix_layout, transr, uplo, n, get_cste_ptr(ap), get_ptr(arf));
            }
        } break;
        case dtpttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary ap; c_binary arf;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &ap, &arf))
            ){
                LAPACKE_dtpttf(matrix_layout, transr, uplo, n, get_cste_ptr(ap), get_ptr(arf));
            }
        } break;
        case ctpttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary ap; c_binary arf;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &ap, &arf))
            ){
                LAPACKE_ctpttf(matrix_layout, transr, uplo, n, get_cste_ptr(ap), get_ptr(arf));
            }
        } break;
        case ztpttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary ap; c_binary arf;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &ap, &arf))
            ){
                LAPACKE_ztpttf(matrix_layout, transr, uplo, n, get_cste_ptr(ap), get_ptr(arf));
            }
        } break;
        case stpttr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &a, &lda))
            ){
                LAPACKE_stpttr(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(a), lda);
            }
        } break;
        case dtpttr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &a, &lda))
            ){
                LAPACKE_dtpttr(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(a), lda);
            }
        } break;
        case ctpttr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &a, &lda))
            ){
                LAPACKE_ctpttr(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(a), lda);
            }
        } break;
        case ztpttr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &a, &lda))
            ){
                LAPACKE_ztpttr(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(a), lda);
            }
        } break;
        case strcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary a; int lda; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &a, &lda, &rcond))
            ){
                LAPACKE_strcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(a), lda, get_ptr(rcond));
            }
        } break;
        case dtrcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary a; int lda; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &a, &lda, &rcond))
            ){
                LAPACKE_dtrcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(a), lda, get_ptr(rcond));
            }
        } break;
        case ctrcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary a; int lda; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &a, &lda, &rcond))
            ){
                LAPACKE_ctrcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(a), lda, get_ptr(rcond));
            }
        } break;
        case ztrcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary a; int lda; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &a, &lda, &rcond))
            ){
                LAPACKE_ztrcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(a), lda, get_ptr(rcond));
            }
        } break;
        case strevc: {
            int matrix_layout; char side; char howmny; c_binary select; int n; cste_c_binary t; int ldt; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_strevc(matrix_layout, side, howmny, get_ptr(select), n, get_cste_ptr(t), ldt, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        } break;
        case dtrevc: {
            int matrix_layout; char side; char howmny; c_binary select; int n; cste_c_binary t; int ldt; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_dtrevc(matrix_layout, side, howmny, get_ptr(select), n, get_cste_ptr(t), ldt, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        } break;
        case ctrevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; c_binary t; int ldt; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_ctrevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        } break;
        case ztrevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; c_binary t; int ldt; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_ztrevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        } break;
        case strexc: {
            int matrix_layout; char compq; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary ifst; c_binary ilst;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &compq, &n, &t, &ldt, &q, &ldq, &ifst, &ilst))
            ){
                LAPACKE_strexc(matrix_layout, compq, n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(ifst), get_ptr(ilst));
            }
        } break;
        case dtrexc: {
            int matrix_layout; char compq; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary ifst; c_binary ilst;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &compq, &n, &t, &ldt, &q, &ldq, &ifst, &ilst))
            ){
                LAPACKE_dtrexc(matrix_layout, compq, n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(ifst), get_ptr(ilst));
            }
        } break;
        case ctrexc: {
            int matrix_layout; char compq; int n; c_binary t; int ldt; c_binary q; int ldq; int ifst; int ilst;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_int, e_end}, &matrix_layout, &compq, &n, &t, &ldt, &q, &ldq, &ifst, &ilst))
            ){
                LAPACKE_ctrexc(matrix_layout, compq, n, get_ptr(t), ldt, get_ptr(q), ldq, ifst, ilst);
            }
        } break;
        case ztrexc: {
            int matrix_layout; char compq; int n; c_binary t; int ldt; c_binary q; int ldq; int ifst; int ilst;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_int, e_end}, &matrix_layout, &compq, &n, &t, &ldt, &q, &ldq, &ifst, &ilst))
            ){
                LAPACKE_ztrexc(matrix_layout, compq, n, get_ptr(t), ldt, get_ptr(q), ldq, ifst, ilst);
            }
        } break;
        case strrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_strrfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case dtrrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dtrrfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case ctrrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ctrrfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case ztrrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ztrrfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        } break;
        case strsen: {
            int matrix_layout; char job; char compq; cste_c_binary select; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary wr; c_binary wi; c_binary m; c_binary s; c_binary sep;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &compq, &select, &n, &t, &ldt, &q, &ldq, &wr, &wi, &m, &s, &sep))
            ){
                LAPACKE_strsen(matrix_layout, job, compq, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(wr), get_ptr(wi), get_ptr(m), get_ptr(s), get_ptr(sep));
            }
        } break;
        case dtrsen: {
            int matrix_layout; char job; char compq; cste_c_binary select; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary wr; c_binary wi; c_binary m; c_binary s; c_binary sep;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &compq, &select, &n, &t, &ldt, &q, &ldq, &wr, &wi, &m, &s, &sep))
            ){
                LAPACKE_dtrsen(matrix_layout, job, compq, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(wr), get_ptr(wi), get_ptr(m), get_ptr(s), get_ptr(sep));
            }
        } break;
        case ctrsen: {
            int matrix_layout; char job; char compq; cste_c_binary select; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary w; c_binary m; c_binary s; c_binary sep;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &compq, &select, &n, &t, &ldt, &q, &ldq, &w, &m, &s, &sep))
            ){
                LAPACKE_ctrsen(matrix_layout, job, compq, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(w), get_ptr(m), get_ptr(s), get_ptr(sep));
            }
        } break;
        case ztrsen: {
            int matrix_layout; char job; char compq; cste_c_binary select; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary w; c_binary m; c_binary s; c_binary sep;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &compq, &select, &n, &t, &ldt, &q, &ldq, &w, &m, &s, &sep))
            ){
                LAPACKE_ztrsen(matrix_layout, job, compq, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(w), get_ptr(m), get_ptr(s), get_ptr(sep));
            }
        } break;
        case strsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary t; int ldt; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary sep; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &s, &sep, &mm, &m))
            ){
                LAPACKE_strsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(t), ldt, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(sep), mm, get_ptr(m));
            }
        } break;
        case dtrsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary t; int ldt; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary sep; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &s, &sep, &mm, &m))
            ){
                LAPACKE_dtrsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(t), ldt, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(sep), mm, get_ptr(m));
            }
        } break;
        case ctrsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary t; int ldt; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary sep; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &s, &sep, &mm, &m))
            ){
                LAPACKE_ctrsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(t), ldt, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(sep), mm, get_ptr(m));
            }
        } break;
        case ztrsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary t; int ldt; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary sep; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &s, &sep, &mm, &m))
            ){
                LAPACKE_ztrsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(t), ldt, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(sep), mm, get_ptr(m));
            }
        } break;
        case strsyl: {
            int matrix_layout; char trana; char tranb; int isgn; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; c_binary scale;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trana, &tranb, &isgn, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &scale))
            ){
                LAPACKE_strsyl(matrix_layout, trana, tranb, isgn, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_ptr(scale));
            }
        } break;
        case dtrsyl: {
            int matrix_layout; char trana; char tranb; int isgn; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; c_binary scale;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trana, &tranb, &isgn, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &scale))
            ){
                LAPACKE_dtrsyl(matrix_layout, trana, tranb, isgn, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_ptr(scale));
            }
        } break;
        case ctrsyl: {
            int matrix_layout; char trana; char tranb; int isgn; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; c_binary scale;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trana, &tranb, &isgn, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &scale))
            ){
                LAPACKE_ctrsyl(matrix_layout, trana, tranb, isgn, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_ptr(scale));
            }
        } break;
        case ztrsyl: {
            int matrix_layout; char trana; char tranb; int isgn; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; c_binary scale;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trana, &tranb, &isgn, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &scale))
            ){
                LAPACKE_ztrsyl(matrix_layout, trana, tranb, isgn, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_ptr(scale));
            }
        } break;
        case strtri: {
            int matrix_layout; char uplo; char diag; int n; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &diag, &n, &a, &lda))
            ){
                LAPACKE_strtri(matrix_layout, uplo, diag, n, get_ptr(a), lda);
            }
        } break;
        case dtrtri: {
            int matrix_layout; char uplo; char diag; int n; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &diag, &n, &a, &lda))
            ){
                LAPACKE_dtrtri(matrix_layout, uplo, diag, n, get_ptr(a), lda);
            }
        } break;
        case ctrtri: {
            int matrix_layout; char uplo; char diag; int n; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &diag, &n, &a, &lda))
            ){
                LAPACKE_ctrtri(matrix_layout, uplo, diag, n, get_ptr(a), lda);
            }
        } break;
        case ztrtri: {
            int matrix_layout; char uplo; char diag; int n; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &diag, &n, &a, &lda))
            ){
                LAPACKE_ztrtri(matrix_layout, uplo, diag, n, get_ptr(a), lda);
            }
        } break;
        case strtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_strtrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case dtrtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dtrtrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case ctrtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_ctrtrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case ztrtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_ztrtrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        } break;
        case strttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary a; int lda; c_binary arf;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a, &lda, &arf))
            ){
                LAPACKE_strttf(matrix_layout, transr, uplo, n, get_cste_ptr(a), lda, get_ptr(arf));
            }
        } break;
        case dtrttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary a; int lda; c_binary arf;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a, &lda, &arf))
            ){
                LAPACKE_dtrttf(matrix_layout, transr, uplo, n, get_cste_ptr(a), lda, get_ptr(arf));
            }
        } break;
        case ctrttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary a; int lda; c_binary arf;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a, &lda, &arf))
            ){
                LAPACKE_ctrttf(matrix_layout, transr, uplo, n, get_cste_ptr(a), lda, get_ptr(arf));
            }
        } break;
        case ztrttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary a; int lda; c_binary arf;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a, &lda, &arf))
            ){
                LAPACKE_ztrttf(matrix_layout, transr, uplo, n, get_cste_ptr(a), lda, get_ptr(arf));
            }
        } break;
        case strttp: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ap))
            ){
                LAPACKE_strttp(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(ap));
            }
        } break;
        case dtrttp: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ap))
            ){
                LAPACKE_dtrttp(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(ap));
            }
        } break;
        case ctrttp: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ap))
            ){
                LAPACKE_ctrttp(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(ap));
            }
        } break;
        case ztrttp: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ap))
            ){
                LAPACKE_ztrttp(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(ap));
            }
        } break;
        case stzrzf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_stzrzf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case dtzrzf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dtzrzf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case ctzrzf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_ctzrzf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case ztzrzf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_ztzrzf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        } break;
        case cungbr: {
            int matrix_layout; char vect; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &vect, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cungbr(matrix_layout, vect, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case zungbr: {
            int matrix_layout; char vect; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &vect, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zungbr(matrix_layout, vect, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case cunghr: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_cunghr(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case zunghr: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_zunghr(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case cunglq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cunglq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case zunglq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zunglq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case cungql: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cungql(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case zungql: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zungql(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case cungqr: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cungqr(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case zungqr: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zungqr(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case cungrq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cungrq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case zungrq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zungrq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case cungtr: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &tau))
            ){
                LAPACKE_cungtr(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case zungtr: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &tau))
            ){
                LAPACKE_zungtr(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(tau));
            }
        } break;
        case cunmbr: {
            int matrix_layout; char vect; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmbr(matrix_layout, vect, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case zunmbr: {
            int matrix_layout; char vect; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmbr(matrix_layout, vect, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case cunmhr: {
            int matrix_layout; char side; char trans; int m; int n; int ilo; int ihi; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &ilo, &ihi, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmhr(matrix_layout, side, trans, m, n, ilo, ihi, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case zunmhr: {
            int matrix_layout; char side; char trans; int m; int n; int ilo; int ihi; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &ilo, &ihi, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmhr(matrix_layout, side, trans, m, n, ilo, ihi, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case cunmlq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmlq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case zunmlq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmlq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case cunmql: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmql(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case zunmql: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmql(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case cunmqr: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmqr(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case zunmqr: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmqr(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case cunmrq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmrq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case zunmrq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmrq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case cunmrz: {
            int matrix_layout; char side; char trans; int m; int n; int k; int l; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &l, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmrz(matrix_layout, side, trans, m, n, k, l, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case zunmrz: {
            int matrix_layout; char side; char trans; int m; int n; int k; int l; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &l, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmrz(matrix_layout, side, trans, m, n, k, l, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case cunmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case zunmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case cupgtr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary tau; c_binary q; int ldq;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &tau, &q, &ldq))
            ){
                LAPACKE_cupgtr(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(q), ldq);
            }
        } break;
        case zupgtr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary tau; c_binary q; int ldq;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &tau, &q, &ldq))
            ){
                LAPACKE_zupgtr(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(q), ldq);
            }
        } break;
        case cupmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary ap; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &ap, &tau, &c, &ldc))
            ){
                LAPACKE_cupmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        case zupmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary ap; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &ap, &tau, &c, &ldc))
            ){
                LAPACKE_zupmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(c), ldc);
            }
        } break;
        default:
            error = ERROR_NO_BLAS;
            debug_write("Error: blas %s of hash %u does not exist.\n", name, hash(name));
        break;
    }

    switch(error){
        case ERROR_NO_BLAS:
            return enif_raise_exception(env, enif_make_atom(env, "Unknown blas."));
        case ERROR_NONE:
            return !result? enif_make_atom(env, "ok"): result;
        break;
        case 1 ... 19:
            char buff[50];
            sprintf(buff, "Could not translate argument %i.", error - 1);
            return enif_raise_exception(env, enif_make_atom(env, buff));
        break;
        case ERROR_SIGSEV:
            return enif_raise_exception(env, enif_make_atom(env, "Array overflow."));
        break;
        case ERROR_N_ARG:
            return enif_raise_exception(env, enif_make_atom(env, "Invalid number of arguments."));
        break;

        default:
            return enif_make_badarg(env);
        break;
    }
}


ERL_NIF_TERM hash_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    int max_len = 50;
    char name[max_len];

    if(!enif_get_atom(env, argv[0], name, max_len-1, ERL_NIF_LATIN1)){
        return enif_make_badarg(env);
    }

    unsigned long h = hash(name);

    return enif_make_uint64(env, h);
}

int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info){
    c_binary_resource = enif_open_resource_type(env, "c_binary", "c_binary_resource", NULL, ERL_NIF_RT_CREATE, NULL);

    atomRowMajor    = enif_make_atom(env, "blasRowMajor");
    atomColMajor    = enif_make_atom(env, "blasColMajor");
    atomNoTrans     = enif_make_atom(env, "blasNoTrans");
    atomTrans       = enif_make_atom(env, "blasTrans");
    atomConjTrans   = enif_make_atom(env, "blasConjTrans");
    atomUpper       = enif_make_atom(env, "blasUpper");
    atomLower       = enif_make_atom(env, "blasLower");
    atomNonUnit     = enif_make_atom(env, "blasNonUnit");
    atomUnit        = enif_make_atom(env, "blasUnit");
    atomLeft        = enif_make_atom(env, "blasLeft");
    atomRight       = enif_make_atom(env, "blasRight");

    return 0;
}

ErlNifFunc nif_funcs[] = { 
    {"new_nif", 1, new},
    {"copy_nif", 2, copy},
    {"bin_nif", 2, to_binary},
    {"hash", 1, hash_nif},

    {"dirty_unwrapper", 1, unwrapper, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"clean_unwrapper", 1, unwrapper, 0}
};


ERL_NIF_INIT(blas, nif_funcs, load, NULL, NULL, NULL)