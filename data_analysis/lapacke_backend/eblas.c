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
            case e_uint:
                i_dest = va_arg(valist, int*);
                valid = enif_get_int(env, terms[curr], i_dest) && *i_dest >=0;
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
                else if (enif_is_identical(terms[curr], atomN))        *i_dest = CblasNoTrans;
                else if (enif_is_identical(terms[curr], atomT))        *i_dest = CblasTrans;
                else if (enif_is_identical(terms[curr], atomC))        *i_dest = CblasConjTrans;
                else valid = 0;
            break;

            case e_uplo: 
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomUpper)) *i_dest = CblasUpper;
                else if (enif_is_identical(terms[curr], atomLower)) *i_dest = CblasLower;
                else if (enif_is_identical(terms[curr], atomU    )) *i_dest = CblasUpper;
                else if (enif_is_identical(terms[curr], atomL    )) *i_dest = CblasLower;
                else valid = 0;
            break;

            case e_diag: 
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomNonUnit)) *i_dest = CblasNonUnit;
                else if (enif_is_identical(terms[curr], atomUnit))    *i_dest = CblasUnit;
                else if (enif_is_identical(terms[curr], atomN))       *i_dest = CblasNonUnit;
                else if (enif_is_identical(terms[curr], atomU))       *i_dest = CblasUnit;
                else valid = 0;
                break;

            case e_side:
                i_dest = va_arg(valist, int*);
                if      (enif_is_identical(terms[curr], atomLeft))  *i_dest = CblasLeft;
                else if (enif_is_identical(terms[curr], atomRight)) *i_dest = CblasRight;
                else if (enif_is_identical(terms[curr], atomL))     *i_dest = CblasLeft;
                else if (enif_is_identical(terms[curr], atomR))     *i_dest = CblasRight;
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

    size_in_bytes type = pick_size(hash_name);

    ERL_NIF_TERM result = 0;

    int timeslice;
    if(!enif_get_int(env, argv[1], &timeslice))
        return enif_make_badarg(env);

    if (timeslice < 0){
        timeslice = 100;
    }
    enif_consume_timeslice(env, timeslice);

    switch(hash_name){
         case saxpy: case daxpy: case caxpy: case zaxpy: {
            int n; cste_c_binary alpha; cste_c_binary x; int incx; c_binary y; int incy;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &n, &alpha, &x, &incx, &y, &incy))
                && !(error = in_cste_bounds(type, 1, 1, alpha)) 
                && !(error = in_cste_bounds(type, n, incx, x))  &&
                 !(error = in_bounds(type, n, incy, y))
            ){
                switch(hash_name){
                    case saxpy: cblas_saxpy(n,  get_cste_float(alpha), get_cste_ptr(x), incx, get_ptr(y), incy); break;
                    case daxpy: cblas_daxpy(n, get_cste_double(alpha), get_cste_ptr(x), incx, get_ptr(y), incy); break;
                    case caxpy: cblas_caxpy(n,           get_cste_ptr(alpha), get_cste_ptr(x), incx, get_ptr(y), incy); break;
                    case zaxpy: cblas_zaxpy(n,           get_cste_ptr(alpha), get_cste_ptr(x), incx, get_ptr(y), incy); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
            }


        break;}

        case scopy: case dcopy: case ccopy: case zcopy:  {
            int n;  cste_c_binary x; int incx; c_binary y; int incy;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &n, &x, &incx, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y))
            ){
                switch(hash_name){
                    case scopy: cblas_scopy(n, get_cste_ptr(x), incx, get_ptr(y), incy); break;
                    case dcopy: cblas_dcopy(n, get_cste_ptr(x), incx, get_ptr(y), incy); break;
                    case ccopy: cblas_ccopy(n, get_cste_ptr(x), incx, get_ptr(y), incy); break;
                    case zcopy: cblas_zcopy(n, get_cste_ptr(x), incx, get_ptr(y), incy); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }

        break;}

        case sswap: case dswap: case cswap: case zswap:  {
            int n;  c_binary x; int incx; c_binary y; int incy;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &n, &x, &incx, &y, &incy))
                && !(error = in_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y))
            ){
                switch(hash_name){
                    case sswap: cblas_sswap(n, get_ptr(x), incx, get_ptr(y), incy); break;
                    case dswap: cblas_dswap(n, get_ptr(x), incx, get_ptr(y), incy); break;
                    case cswap: cblas_cswap(n, get_ptr(x), incx, get_ptr(y), incy); break;
                    case zswap: cblas_zswap(n, get_ptr(x), incx, get_ptr(y), incy); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
            
        break;}

        case sscal: case dscal: case cscal: case zscal: case csscal: case zdscal:  {
            int n;  cste_c_binary alpha; c_binary x; int incx;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_ptr, e_int, e_end}, &n, &alpha, &x, &incx))
                && !(error = in_cste_bounds(type, 1, 1, alpha) ) 
                && !(error = in_bounds(type, n, incx, x))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
            ){
                switch(hash_name){
                    case sscal:  cblas_sscal(n,  *(float*) get_cste_ptr(alpha), get_ptr(x), incx); break;
                    case dscal:  cblas_dscal(n,get_cste_double(alpha), get_ptr(x), incx); break;
                    case cscal:  cblas_cscal(n,            get_cste_ptr(alpha), get_ptr(x), incx); break;
                    case zscal:  cblas_zscal(n,            get_cste_ptr(alpha), get_ptr(x), incx); break;
                    case csscal: cblas_sscal(n,  *(float*) get_cste_ptr(alpha), get_ptr(x), incx); break;
                    case zdscal: cblas_dscal(n,get_cste_double(alpha), get_ptr(x), incx); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
            
        break;}

        case sdot: case ddot: case dsdot: case cdotu: case zdotu: case cdotc: case zdotc: {
            cste_c_binary dot_result;

            int n;  cste_c_binary x; int incx; cste_c_binary y; int incy;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_end}, &n, &x, &incx, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x) ) && !(error = in_cste_bounds(type, n, incy, y))
            ){ 
                switch(hash_name){
                    case sdot:                   double f_result  = cblas_sdot (n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &f_result);  break;
                    case ddot:                   double d_result  = cblas_ddot (n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &d_result);  break;
                    case dsdot:                  double ds_result = cblas_dsdot(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &ds_result); break;
                    case cdotu: openblas_complex_float  c_result  = cblas_cdotu(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_float_complex,  (unsigned char*) &c_result);  break;
                    case zdotu: openblas_complex_double z_result  = cblas_zdotu(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double_complex, (unsigned char*) &z_result);  break;
                    case cdotc: openblas_complex_float  cd_result = cblas_cdotc(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_float_complex,  (unsigned char*) &cd_result); break;
                    case zdotc: openblas_complex_double zd_result = cblas_zdotc(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double_complex, (unsigned char*) &zd_result); break;
                    default: error = ERROR_NOT_FOUND; break;
                }

                result = cste_c_binary_to_term(env, dot_result);
            }
            
        break;}

        case sdsdot: {
            cste_c_binary dot_result;

            int n;  cste_c_binary b; cste_c_binary x; int incx; cste_c_binary y; int incy;
            size_in_bytes type = s_bytes;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_end}, &n, &b, &x, &incx, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x) ) && !(error = in_cste_bounds(type, n, incy, y))
            ){
                double f_result  = cblas_sdsdot (n, *(float*) get_cste_ptr(b), get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &f_result);
                result = cste_c_binary_to_term(env, dot_result);
                
            }
        break;}

        case snrm2: case dnrm2: case scnrm2: case dznrm2:
        case ssum: case dsum: case scsum: case dzsum: {
            cste_c_binary u_result;
            double d_result;
            int i_result;

            int n;  cste_c_binary x; int incx;

            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_end}, &n, &x, &incx))
                && !(error = in_cste_bounds(type, n, incx, x))
            ){
                switch(hash_name){
                    case snrm2:  d_result  = cblas_snrm2 (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case dnrm2:  d_result  = cblas_dnrm2 (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case scnrm2: d_result  = cblas_scnrm2(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case dznrm2: d_result  = cblas_dznrm2(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    
                    case dsum:  d_result  = cblas_dsum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case ssum:  d_result  = cblas_ssum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case scsum: d_result  = cblas_scsum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case dzsum: d_result  = cblas_dzsum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    
                    case dasum:  d_result  = cblas_dasum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case sasum:  d_result  = cblas_sasum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case scasum: d_result  = cblas_scasum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case dzasum: d_result  = cblas_dzasum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    
                    case isamax: i_result  = cblas_isamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case idamax: i_result  = cblas_idamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case icamax: i_result  = cblas_icamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case izamax: i_result  = cblas_izamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    case isamin: i_result  = cblas_isamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case idamin: i_result  = cblas_idamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case icamin: i_result  = cblas_icamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case izamin: i_result  = cblas_izamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    case ismax: i_result  = cblas_ismax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case idmax: i_result  = cblas_idmax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case icmax: i_result  = cblas_icmax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case izmax: i_result  = cblas_izmax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    case ismin: i_result  = cblas_ismin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case idmin: i_result  = cblas_idmin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case icmin: i_result  = cblas_icmin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case izmin: i_result  = cblas_izmin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                result = cste_c_binary_to_term(env, u_result);
            }
            
        break;}

        case srot: case drot: case csrot: case zdrot:  {
            int n;  c_binary x; int incx; c_binary y; int incy; cste_c_binary c; cste_c_binary s;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_end}, &n, &x, &incx, &y, &incy, &c, &s))
                && !(error = in_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y))
            ){
                switch(hash_name){
                    case srot:  cblas_srot(n, get_ptr(x),  incx, get_ptr(y), incy, get_cste_float(c), get_cste_float(s)); break;
                    case drot:  cblas_drot(n, get_ptr(x),  incx, get_ptr(y), incy, get_cste_double(c), get_cste_double(s)); break;
                    case csrot: cblas_csrot(n, get_ptr(x), incx, get_ptr(y), incy, get_cste_float(c), get_cste_float(s)); break;
                    case zdrot: cblas_zdrot(n, get_ptr(x), incx, get_ptr(y), incy, get_cste_double(c), get_cste_double(s)); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
            
        break;}

        case srotg: case drotg: case crotg: case zrotg:  {
            c_binary a; c_binary b; c_binary c; c_binary s;

            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &a, &b, &c, &s))
                && !(error = in_bounds(type, 1, 1, a)) && !(error = in_bounds(type, 1, 1, b)) && !(error = in_bounds(type, 1, 1, c)) && !(error = in_bounds(type, 1, 1, s))
            ){
                switch(hash_name){
                    case srotg: cblas_srotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s)); break;
                    case drotg: cblas_drotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s)); break;
                    case crotg: cblas_crotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s)); break;
                    case zrotg: cblas_zrotg(get_ptr(a), get_ptr(b), get_ptr(c), get_ptr(s)); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
            
        break;}

        case srotm: case drotm:  {
            int n; c_binary x; int incx; c_binary y; int incy; cste_c_binary param;

            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &n, &x, &incx, &y, &incy, &param))
                && !(error = in_bounds(type, n, incx, x)) && !(error = in_bounds(type, n, incy, y)) && !(error = in_cste_bounds(type, 5, 1, param))
            ){
                switch(hash_name){
                    case srotm: cblas_srotm(n, get_ptr(x), incx, get_ptr(y), incy, get_cste_ptr(param)); break;
                    case drotm: cblas_srotm(n, get_ptr(x), incx, get_ptr(y), incy, get_cste_ptr(param)); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
            }
            
        break;}

        case srotmg: case drotmg:  {
            c_binary d1; c_binary d2; c_binary b1; cste_c_binary b2; c_binary param;
            

            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_ptr, e_ptr, e_ptr, e_cste_ptr, e_ptr, e_end}, &d1, &d2, &b1, &b2, &param))
                && !(error = in_bounds(type, 1, 1, d1)) && !(error = in_bounds(type, 1, 1, d2)) && !(error = in_bounds(type, 1, 1, b1)) && !(error = in_cste_bounds(type, 1, 1, b2)) && !(error = in_bounds(type, 5, 1, param))
            ){
                switch(hash_name){
                    case srotmg: cblas_srotmg(get_ptr(d1), get_ptr(d2), get_ptr(b1), get_cste_float(b2),  get_ptr(param)); break;
                    case drotmg: cblas_srotmg(get_ptr(d1), get_ptr(d2), get_ptr(b1), get_cste_double(b2),  get_ptr(param)); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
            
            }
            
        break;}

        // BLAS LEVEL 2
        // GENERAL MATRICES

        case sgemv: case dgemv: case cgemv: case zgemv: {
            int layout; int trans; int m; int n; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;

            if( !(error = narg == 12?0:ERROR_N_ARG)
                && ! (error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_uint, e_uint, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
                                                        &layout, &trans, &m, &n, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x)) 
                && !(error = in_bounds(type, n, incy, y))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_cste_bounds(type, 1, 1, beta))
                && !(error = in_cste_bounds(type, leading_dim(layout, trans, m, n), lda, a))
            ){
                switch(hash_name){
                    case sgemv: cblas_sgemv(layout, trans, m, n,  get_cste_float(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,  get_cste_float(beta), get_ptr(y), incy); break;
                    case dgemv: cblas_dgemv(layout, trans, m, n, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx, get_cste_double(beta), get_ptr(y), incy); break;
                    case cgemv: cblas_cgemv(layout, trans, m, n,    get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,    get_cste_ptr(beta), get_ptr(y), incy); break;
                    case zgemv: cblas_zgemv(layout, trans, m, n,    get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,     get_cste_ptr(beta), get_ptr(y), incy); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
        break;}

        case sgbmv: case dgbmv: case cgbmv: case zgbmv: {
            int layout; int trans; int m; int n; int kl; int ku; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;

            if( !(error = narg == 14?0:ERROR_N_ARG)
                && ! (error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_uint, e_uint, e_uint, e_uint, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
                                                        &layout, &trans, &m, &n, &kl, &ku, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x)) 
                && !(error = in_bounds(type, n, incy, y))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_cste_bounds(type, 1, 1, beta))
                && !(error = in_cste_bounds(type, lda, n, a))

            ){
                switch(hash_name){
                    case sgbmv: cblas_sgbmv(layout, trans, m, n, kl, ku,  get_cste_float(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,  get_cste_float(beta), get_ptr(y), incy); break;
                    case dgbmv: cblas_dgbmv(layout, trans, m, n, kl, ku, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx, get_cste_double(beta), get_ptr(y), incy); break;
                    case cgbmv: cblas_cgbmv(layout, trans, m, n, kl, ku,    get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,    get_cste_ptr(beta), get_ptr(y), incy); break;
                    case zgbmv: cblas_zgbmv(layout, trans, m, n, kl, ku,    get_cste_ptr(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,    get_cste_ptr(beta), get_ptr(y), incy); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
        break;}

        case ssbmv: case dsbmv: {
            int layout; int uplo; int m; int n; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;
            
            if( !(error = narg == 12?0:ERROR_N_ARG)
                && ! (error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_uint, e_uint, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
                                                        &layout, &uplo, &m, &n, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy))
                &&!(error = in_cste_bounds(type, n, incx, x)) 
                && !(error = in_bounds(type, n, incy, y))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_cste_bounds(type, 1, 1, beta))
                && !(error = in_cste_bounds(type, lda, n, a))
            ){
                switch(hash_name){
                    case ssbmv: cblas_ssbmv(layout, uplo, m, n, get_cste_float(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx, get_cste_float(beta), get_ptr(y), incy); break;
                    case dsbmv: cblas_dsbmv(layout, uplo, m, n, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_ptr(x), incx,get_cste_double(beta), get_ptr(y), incy); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
        break;}

        case strmv: case dtrmv: case ctrmv: case ztrmv: {
            int order; int uplo; int transa; int diag; int n; cste_c_binary a; int lda; c_binary x; int incx;

            if( !(error = test_n_arg(narg, 9))
                && ! (error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end},
                                                        &order, &uplo, &transa, &diag, &n, &a, &lda, &x, &incx))
                &&!(error = in_bounds(type, n, incx, x))
                && !(error = in_cste_bounds(type, lda, n, a))
            ){
                switch(hash_name){
                    case strmv: cblas_strmv(order, uplo, transa, diag, n, get_cste_ptr(a), lda, get_ptr(x), incx); break;
                    case dtrmv: cblas_dtrmv(order, uplo, transa, diag, n, get_cste_ptr(a), lda, get_ptr(x), incx); break;
                    case ctrmv: cblas_ctrmv(order, uplo, transa, diag, n, get_cste_ptr(a), lda, get_ptr(x), incx); break;
                    case ztrmv: cblas_ztrmv(order, uplo, transa, diag, n, get_cste_ptr(a), lda, get_ptr(x), incx); break;
                    default: error = ERROR_NOT_FOUND; break;
                }
            }
                
        break;}

        //=======================

        case strsv: case dtrsv: case ctrsv: case ztrsv: {
			int order; int transa; int uplo; int diag; int n; cste_c_binary a; int lda; c_binary x; int incx;

			if(!(error = test_n_arg(narg, 9))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end},
			                                     &order, &uplo, &transa, &diag, &n, &a, &lda, &x, &incx))
            && !(error = in_bounds(type, n, incx, x))
            && !(error = in_cste_bounds(type, lda, n, a))
			)
            {
                switch(hash_name){
                    case strsv:	cblas_strsv(order, uplo, transa, diag, n,  get_cste_ptr(a), lda,  get_ptr(x), incx); break;
                    case dtrsv:	cblas_dtrsv(order, uplo, transa, diag, n,  get_cste_ptr(a), lda,  get_ptr(x), incx); break;
                    case ctrsv:	cblas_ctrsv(order, uplo, transa, diag, n,  get_cste_ptr(a), lda,  get_ptr(x), incx); break;
                    case ztrsv:	cblas_ztrsv(order, uplo, transa, diag, n,  get_cste_ptr(a), lda,  get_ptr(x), incx); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

        case strmm: case dtrmm: case ctrmm: case ztrmm: {
            int order; int side; int uplo; int transa; int diag; int m; int n; cste_c_binary alpha; cste_c_binary a; int lda; c_binary b; int ldb;

			if(!(error = test_n_arg(narg, 12))
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end},
                                                    &order, &side, &uplo, &transa, &diag, &m, &n, &alpha, &a, &lda, &b, &ldb))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_cste_bounds(type, lda, side == CblasLeft? m:n, a))
			){
                switch(hash_name){
                    case strmm:	cblas_strmm(order, side, uplo, transa, diag, m, n,  get_cste_float(alpha), get_cste_ptr(a), lda,  get_ptr(b), ldb); break;
                    case dtrmm:	cblas_dtrmm(order, side, uplo, transa, diag, m, n,  get_cste_double(alpha), get_cste_ptr(a), lda,  get_ptr(b), ldb); break;
                    case ctrmm:	cblas_ctrmm(order, side, uplo, transa, diag, m, n,  get_cste_ptr(alpha), get_cste_ptr(a), lda,  get_ptr(b), ldb); break;
                    case ztrmm:	cblas_ztrmm(order, side, uplo, transa, diag, m, n,  get_cste_ptr(alpha), get_cste_ptr(a), lda,  get_ptr(b), ldb); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}


		case sger: case dger: case cgeru: case cgerc: case zgeru: case zgerc: {
			int order; int m; int n; cste_c_binary alpha; cste_c_binary x; int incx; cste_c_binary y; int incy; c_binary a; int lda;

			if(!(error = test_n_arg(narg, 10))
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end},
                                                    &order, &m, &n, &alpha, &x, &incx, &y, &incy, &a, &lda))
                && !(error = in_cste_bounds(type, n, incx, x))
                && !(error = in_cste_bounds(type, n, incy, y))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_bounds(type, leading_dim(order, CblasNoTrans, m, n), lda, a))
			){
                switch(hash_name){
                    case sger:	cblas_sger(order, m, n,  get_cste_float(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(a), lda); break;
                    case dger:	cblas_dger(order, m, n,  get_cste_double(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(a), lda); break;
                    case cgeru:	cblas_cgeru(order, m, n,  get_cste_ptr(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(a), lda); break;
                    case cgerc:	cblas_cgerc(order, m, n,  get_cste_ptr(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(a), lda); break;
                    case zgeru:	cblas_zgeru(order, m, n,  get_cste_ptr(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(a), lda); break;
                    case zgerc:	cblas_zgerc(order, m, n,  get_cste_ptr(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(a), lda); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case sgemm: case dgemm: case cgemm: case cgemm3m: case zgemm: case zgemm3m: {
			int order; int transa; int transb; int m; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary beta; c_binary c; int ldc;

			if(!(error = test_n_arg(narg, 14))
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_transpose, e_transpose, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
                                                    &order, &transa, &transb, &m, &n, &k, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_cste_bounds(type, 1, 1, beta))
                && !(error = in_cste_bounds(type, lda, leading_dim(order, transa, m, n), a))
                && !(error = in_cste_bounds(type, ldb, leading_dim(order, transb, m, n), b))
                && !(error = in_bounds(type, ldc, leading_dim(order, CblasNoTrans, m, n), c))
            ){
			switch(hash_name){
				case sgemm:	  cblas_sgemm(  order, transa, transb, m, n, k,  get_cste_float(alpha),   get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,  get_cste_float(beta),  get_ptr(c), ldc); break;
				case dgemm:	  cblas_dgemm(  order, transa, transb, m, n, k,  get_cste_double(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb, get_cste_double(beta),  get_ptr(c), ldc); break;
				case cgemm:	  cblas_cgemm(  order, transa, transb, m, n, k,  get_cste_ptr(alpha),     get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,    get_cste_ptr(beta),  get_ptr(c), ldc); break;
				case cgemm3m: cblas_cgemm3m(order, transa, transb, m, n, k,  get_cste_ptr(alpha),     get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,    get_cste_ptr(beta),  get_ptr(c), ldc); break;
				case zgemm:	  cblas_zgemm(  order, transa, transb, m, n, k,  get_cste_ptr(alpha),     get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,    get_cste_ptr(beta),  get_ptr(c), ldc); break;
				case zgemm3m: cblas_zgemm3m(order, transa, transb, m, n, k,  get_cste_ptr(alpha),     get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,    get_cste_ptr(beta),  get_ptr(c), ldc); break;

				default: error = ERROR_NOT_FOUND; break;
			}
            
    }
		break;}

		case stbmv: case dtbmv: case ctbmv: case ztbmv: {
			int order; int transa; int uplo; int diag; int n; int k; cste_c_binary a; int lda; c_binary x; int incx;

			if(!(error = test_n_arg(narg, 10))
			    && !(error = translate(env, elements, (etypes[]) {e_layout,  e_uplo, e_transpose,  e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end},
			                                     &order, &uplo, &transa, &diag, &n, &k, &a, &lda, &x, &incx))
                && !(error = in_bounds(type, n, incx, x))
                && !(error = in_cste_bounds(type, lda, n, a))
			){
                switch(hash_name){
                    case stbmv:	cblas_stbmv(order, uplo, transa, diag, n, k,  get_cste_ptr(a), lda,  get_ptr(x), incx); break;
                    case dtbmv:	cblas_dtbmv(order, uplo, transa, diag, n, k,  get_cste_ptr(a), lda,  get_ptr(x), incx); break;
                    case ctbmv:	cblas_ctbmv(order, uplo, transa, diag, n, k,  get_cste_ptr(a), lda,  get_ptr(x), incx); break;
                    case ztbmv:	cblas_ztbmv(order, uplo, transa, diag, n, k,  get_cste_ptr(a), lda,  get_ptr(x), incx); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case stbsv: case dtbsv: case ctbsv: case ztbsv: {
			int order; int transa; int uplo; int diag; int n; int k; cste_c_binary a;  int lda; c_binary x; int incx;

			if(!(error = test_n_arg(narg, 10))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end},
			                                     &order, &uplo, &transa, &diag, &n, &k, &a, &lda, &x, &incx))
            && !(error = in_bounds(type, n, incx, x))
            && !(error = in_cste_bounds(type, lda, n, a))
			){
                switch(hash_name){
                    case stbsv:	cblas_stbsv(order, uplo, transa, diag, n, k, get_cste_ptr(a), lda, get_ptr(x), incx); break;
                    case dtbsv:	cblas_dtbsv(order, uplo, transa, diag, n, k, get_cste_ptr(a), lda, get_ptr(x), incx); break;
                    case ctbsv:	cblas_ctbsv(order, uplo, transa, diag, n, k, get_cste_ptr(a), lda, get_ptr(x), incx); break;
                    case ztbsv:	cblas_ztbsv(order, uplo, transa, diag, n, k, get_cste_ptr(a), lda, get_ptr(x), incx); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case stpmv: case dtpmv: case ctpmv: case ztpmv: {
			int order; int transa; int uplo; int diag; int n; cste_c_binary ap; c_binary x; int incx;

			if(!(error = test_n_arg(narg, 8))
			&& !(error = translate(env, elements, (etypes[]) {e_layout,  e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order, &uplo, &transa, &diag, &n, &ap, &x, &incx))
            && !(error = in_bounds(type, n, incx, x))
            && !(error = in_cste_bounds(type, (n*(n+1))/2, 1, ap)
            )){
                switch(hash_name){
                    case stpmv:	cblas_stpmv(order,  uplo, transa, diag, n,  get_cste_ptr(ap),  get_ptr(x), incx); break;
                    case dtpmv:	cblas_dtpmv(order,  uplo, transa, diag, n,  get_cste_ptr(ap),  get_ptr(x), incx); break;
                    case ctpmv:	cblas_ctpmv(order,  uplo, transa, diag, n,  get_cste_ptr(ap),  get_ptr(x), incx); break;
                    case ztpmv:	cblas_ztpmv(order,  uplo, transa, diag, n,  get_cste_ptr(ap),  get_ptr(x), incx); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
                			}
		break;}

		case stpsv: case dtpsv: case ctpsv: case ztpsv: {
			int order; int transa; int uplo; int diag; int n; cste_c_binary ap; c_binary x; int incx;

			if(!(error = test_n_arg(narg, 8))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_diag, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order, &uplo, &transa, &diag, &n, &ap, &x, &incx))
            && !(error = in_bounds(type, n, incx, x))
            && !(error = in_cste_bounds(type, (n*(n+1))/2, 1, ap)
            )){
                switch(hash_name){
                    case stpsv:	cblas_stpsv(order, uplo, transa, diag, n,  get_cste_ptr(ap),  get_ptr(x), incx); break;
                    case dtpsv:	cblas_dtpsv(order, uplo, transa, diag, n,  get_cste_ptr(ap),  get_ptr(x), incx); break;
                    case ctpsv:	cblas_ctpsv(order, uplo, transa, diag, n,  get_cste_ptr(ap),  get_ptr(x), incx); break;
                    case ztpsv:	cblas_ztpsv(order, uplo, transa, diag, n,  get_cste_ptr(ap),  get_ptr(x), incx); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
                
            }
		break;}

		case ssymv: case dsymv: case chemv: case zhemv: {
			int order; int uplo; int n; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;
			if(
                    !(error = test_n_arg(narg, 11))
			    &&  !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order, &uplo, &n, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy))
                &&  !(error = in_cste_bounds(type, n, incx, x))
                &&  !(error = in_bounds(type, n, incy, y))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_cste_bounds(type, 1, 1, beta))
                && !(error = in_cste_bounds(type, lda, n, a))
			){
                switch(hash_name){
                    case ssymv:	cblas_ssymv(order, uplo, n, get_cste_double(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(x), incx, get_cste_double(beta),  get_ptr(y), incy); break;
                    case dsymv:	cblas_dsymv(order, uplo, n, get_cste_double(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(x), incx, get_cste_double(beta),  get_ptr(y), incy); break;
                    case chemv:	cblas_chemv(order, uplo, n,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(x), incx,  get_cste_ptr(beta),  get_ptr(y), incy); break;
                    case zhemv:	cblas_zhemv(order, uplo, n,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(x), incx,  get_cste_ptr(beta),  get_ptr(y), incy); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }


		break;}

		case sspmv: case dspmv: {
			int order; int uplo; int n; cste_c_binary alpha; cste_c_binary ap; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;

			if(!(error = test_n_arg(narg, 10))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order, &uplo, &n, &alpha, &ap, &x, &incx, &beta, &y, &incy))
            && !(error = in_cste_bounds(type, n, incx, x))
            && !(error = in_bounds(type, n, incx, y))
            && !(error = in_cste_bounds(type, 1, 1, alpha))
            && !(error = in_cste_bounds(type, 1, 1, beta))
            && !(error = in_cste_bounds(type, (n*(n+1))/2, 1, ap))
            ){
                switch(hash_name){
                    case sspmv:	cblas_sspmv(order, uplo, n, get_cste_float(alpha),  get_cste_ptr(ap),  get_cste_ptr(x), incx, get_cste_float(beta),  get_ptr(y), incy); break;
                    case dspmv:	cblas_dspmv(order, uplo, n, get_cste_double(alpha),  get_cste_ptr(ap),  get_cste_ptr(x), incx, get_cste_double(beta),  get_ptr(y), incy); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case sspr: case dspr: case chpr: case zhpr: {
			int order; int uplo; int n; cste_c_binary alpha; cste_c_binary x; int incx; c_binary ap;

			if(!(error = test_n_arg(narg, 7))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_end},
			                                     &order, &uplo, &n, &alpha, &x, &incx, &ap))
            && !(error = in_cste_bounds(type, n, incx, x))
            && !(error = in_cste_bounds(type, 1, 1, alpha))
            && !(error = in_bounds(type, (n*(n+1))/2, 1, ap))
			){
                switch(hash_name){
                    case sspr:	cblas_sspr(order, uplo, n, get_cste_float(alpha),  get_cste_ptr(x), incx,  get_ptr(ap)); break;
                    case dspr:	cblas_dspr(order, uplo, n, get_cste_double(alpha),  get_cste_ptr(x), incx,  get_ptr(ap)); break;
                    case chpr:	cblas_chpr(order, uplo, n, get_cste_float(alpha),  get_cste_ptr(x), incx,  get_ptr(ap)); break;
                    case zhpr:	cblas_zhpr(order, uplo, n, get_cste_double(alpha),  get_cste_ptr(x), incx,  get_ptr(ap)); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case sspr2: case dspr2: case chpr2: case zhpr2: {
			int order; int uplo; int n; cste_c_binary alpha; cste_c_binary x; int incx; cste_c_binary y; int incy; c_binary ap;

			if(!(error = test_n_arg(narg, 9))
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_end},
                                                    &order, &uplo, &n, &alpha, &x, &incx, &y, &incy, &ap))
                && !(error = in_cste_bounds(type, n, incx, x))
                && !(error = in_cste_bounds(type, n, incy, y))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_bounds(type, (n*(n+1))/2, 1, ap))
			){
                switch(hash_name){
                    case sspr2:	cblas_sspr2(order, uplo, n, get_cste_float(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(ap)); break;
                    case dspr2:	cblas_dspr2(order, uplo, n, get_cste_double(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(ap)); break;
                    case chpr2:	cblas_chpr2(order, uplo, n,  get_cste_ptr(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(ap)); break;
                    case zhpr2:	cblas_zhpr2(order, uplo, n,  get_cste_ptr(alpha),  get_cste_ptr(x), incx,  get_cste_ptr(y), incy,  get_ptr(ap)); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case chbmv: case zhbmv: {
			int order; int uplo; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;

			if(!(error = test_n_arg(narg, 12))
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
                                                    &order, &uplo, &n, &k, &alpha, &a, &lda, &x, &incx, &beta, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x))
                && !(error = in_bounds(type, n, incy, y))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_cste_bounds(type, 1, 1, beta))
                && !(error = in_cste_bounds(type, lda, n, a))
			){
                switch(hash_name){
                    case chbmv:	cblas_chbmv(order, uplo, n, k,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(x), incx,  get_cste_ptr(beta),  get_ptr(y), incy); break;
                    case zhbmv:	cblas_zhbmv(order, uplo, n, k,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(x), incx,  get_cste_ptr(beta),  get_ptr(y), incy); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case chpmv: case zhpmv: {
			int order; int uplo; int n; cste_c_binary alpha; cste_c_binary ap; cste_c_binary x; int incx; cste_c_binary beta; c_binary y; int incy;

			if(!(error = test_n_arg(narg, 10))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order, &uplo, &n, &alpha, &ap, &x, &incx, &beta, &y, &incy))
            && !(error = in_cste_bounds(type, n, incx, x))
            && !(error = in_bounds(type, n, incy, y))
            && !(error = in_cste_bounds(type, 1, 1, alpha))
            && !(error = in_cste_bounds(type, 1, 1, beta))
            && !(error = in_cste_bounds(type, (n*(n+1))/2, 1, ap))
			){
                switch(hash_name){
                    case chpmv:	cblas_chpmv(order, uplo, n,  get_cste_ptr(alpha),  get_cste_ptr(ap),  get_cste_ptr(x), incx,  get_cste_ptr(beta),  get_ptr(y), incy); break;
                    case zhpmv:	cblas_zhpmv(order, uplo, n,  get_cste_ptr(alpha),  get_cste_ptr(ap),  get_cste_ptr(x), incx,  get_cste_ptr(beta),  get_ptr(y), incy); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case chemm: case zhemm: {
			int order; int side; int uplo; int m; int n; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary beta; c_binary c; int ldc;

			if(!(error = test_n_arg(narg, 13))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order,  &side, &uplo, &m, &n, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc))
			&& !(error = in_cste_bounds(type, 1, 1, alpha))
            && !(error = in_cste_bounds(type, 1, 1, beta))
            && !(error = in_cste_bounds(type, lda, side==CblasLeft?m:n, a))
            ){
                switch(hash_name){
                    case chemm:	cblas_chemm(order, side, uplo, m, n,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,  get_cste_ptr(beta),  get_ptr(c), ldc); break;
                    case zhemm:	cblas_zhemm(order, side, uplo, m, n,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,  get_cste_ptr(beta),  get_ptr(c), ldc); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

        case ssyr: case dsyr: case cher: case zher: {
            int order; int uplo; int n; cste_c_binary alpha; cste_c_binary x; int incx; c_binary a; int lda;

			if(!(error = test_n_arg(narg, 8))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end},
			                                                  &order, &uplo, &n, &alpha, &x, &incx, &a, &lda))
			&& !(error = in_cste_bounds(type, 1, 1, alpha))
            ){
                switch(hash_name){
                    case ssyr:	cblas_ssyr(order, uplo, n,get_cste_float(alpha), get_cste_ptr(x), incx, get_ptr(a), lda); break;
                    case dsyr:	cblas_dsyr(order, uplo, n,get_cste_double(alpha), get_cste_ptr(x), incx, get_ptr(a), lda); break;
                    case cher:	cblas_cher(order, uplo, n,get_cste_float(alpha), get_cste_ptr(x), incx, get_ptr(a), lda); break;
                    case zher:	cblas_zher(order, uplo, n,get_cste_double(alpha), get_cste_ptr(x), incx, get_ptr(a), lda); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

        case ssyr2: case dsyr2: case cher2: case zher2: {
            int order; int uplo; int n; cste_c_binary alpha; cste_c_binary x; int incx; cste_c_binary y; int incy; c_binary a; int lda;

			if(!(error = test_n_arg(narg, 10))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end},
			                                                  &order, &uplo, &n, &alpha, &x, &incx, &y, &incy, &a, &lda))
			&& !(error = in_cste_bounds(type, 1, 1, alpha))
            ){
                switch(hash_name){
                    case ssyr2:	cblas_ssyr2(order, uplo, n,get_cste_float(alpha), get_cste_ptr(x), incx, get_cste_ptr(y), incy, get_ptr(a), lda); break;
                    case dsyr2:	cblas_dsyr2(order, uplo, n,get_cste_double(alpha), get_cste_ptr(x), incx, get_cste_ptr(y), incy, get_ptr(a), lda); break;
                    case cher2:	cblas_cher2(order, uplo, n,            get_cste_ptr(alpha), get_cste_ptr(x), incx, get_cste_ptr(y), incy, get_ptr(a), lda); break;
                    case zher2:	cblas_zher2(order, uplo, n,            get_cste_ptr(alpha), get_cste_ptr(x), incx, get_cste_ptr(y), incy, get_ptr(a), lda); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case cherk: case zherk: {
			int order; int uplo; int trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c; int ldc;

			if(!(error = test_n_arg(narg, 11))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c, &ldc))
			&& !(error = in_cste_bounds(type, 1, 1, alpha))
            && !(error = in_cste_bounds(type, 1, 1, beta))
            ){
                switch(hash_name){
                    case cherk:	cblas_cherk(order, uplo, trans, n, k, get_cste_float(alpha),  get_cste_ptr(a), lda, get_cste_float(beta),  get_ptr(c), ldc); break;
                    case zherk:	cblas_zherk(order, uplo, trans, n, k, get_cste_double(alpha),  get_cste_ptr(a), lda, get_cste_double(beta),  get_ptr(c), ldc); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case cher2k: case zher2k: {
			int order; int uplo; int trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary beta; c_binary c; int ldc;

			if(!(error = test_n_arg(narg, 13))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order, &uplo, &trans, &n, &k, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc))
			&& !(error = in_cste_bounds(type, 1, 1, alpha))
            && !(error = in_cste_bounds(type, 1, 1, beta))
            ){
                switch(hash_name){
                    case cher2k:	cblas_cher2k(order, uplo, trans, n, k,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb, get_cste_float(beta),  get_ptr(c), ldc); break;
                    case zher2k:	cblas_zher2k(order, uplo, trans, n, k,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb, get_cste_double(beta),  get_ptr(c), ldc); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case ssymm: case dsymm: case csymm: case zsymm: {
			int order; int side; int uplo; int m; int n; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary beta; c_binary c; int ldc;

			if(!(error = test_n_arg(narg, 13))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_side, e_uplo, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order, &side, &uplo, &m, &n, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc))
			&& !(error = in_cste_bounds(type, 1, 1, alpha))
            && !(error = in_cste_bounds(type, 1, 1, beta))
            ){
                switch(hash_name){
                    case ssymm:	cblas_ssymm(order, side, uplo, m, n, get_cste_float(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb, get_cste_float(beta),  get_ptr(c), ldc); break;
                    case dsymm:	cblas_dsymm(order, side, uplo, m, n, get_cste_double(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb, get_cste_double(beta),  get_ptr(c), ldc); break;
                    case csymm:	cblas_csymm(order, side, uplo, m, n,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,  get_cste_ptr(beta),  get_ptr(c), ldc); break;
                    case zsymm:	cblas_zsymm(order, side, uplo, m, n,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,  get_cste_ptr(beta),  get_ptr(c), ldc); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case ssyrk: case dsyrk: case csyrk: case zsyrk: {
			int order; int trans; int uplo; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c; int ldc;

			if(!(error = test_n_arg(narg, 11))
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
                                                    &order, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c, &ldc))
                && !(error = in_cste_bounds(type, 1, 1, alpha))
                && !(error = in_cste_bounds(type, 1, 1, beta))
                && !(error = in_cste_bounds(type, leading_dim(order, trans, n, k), lda, a))
                && !(error = in_cste_bounds(type, n, ldc, a))
            ){
                switch(hash_name){
                    case ssyrk:	cblas_ssyrk(order, uplo, trans,  n, k, get_cste_float(alpha),  get_cste_ptr(a), lda, get_cste_float(beta),  get_ptr(c), ldc); break;
                    case dsyrk:	cblas_dsyrk(order, uplo, trans, n, k, get_cste_double(alpha),  get_cste_ptr(a), lda, get_cste_double(beta),  get_ptr(c), ldc); break;
                    case csyrk:	cblas_csyrk(order, uplo, trans, n, k,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(beta),  get_ptr(c), ldc); break;
                    case zsyrk:	cblas_zsyrk(order, uplo, trans, n, k,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(beta),  get_ptr(c), ldc); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
		break;}

		case ssyr2k: case dsyr2k: case csyr2k: case zsyr2k: {
			int order; int trans; int uplo; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary beta; c_binary c; int ldc;
			
            if(!(error = test_n_arg(narg, 13))
			&& !(error = translate(env, elements, (etypes[]) {e_layout, e_uplo, e_transpose,  e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end},
			                                     &order, &trans, &uplo, &n, &k, &alpha, &a, &lda, &b, &ldb, &beta, &c, &ldc))
			&& !(error = in_cste_bounds(type, 1, 1, alpha))
            && !(error = in_cste_bounds(type, 1, 1, beta))
            && !(error = in_cste_bounds(type, leading_dim(order, trans, k, n), lda, a))
            && !(error = in_cste_bounds(type, leading_dim(order, trans, k, n), lda, a))
            && !(error = in_cste_bounds(type, n, ldc, a))
            ){
                switch(hash_name){
                    case ssyr2k:    cblas_ssyr2k(order, trans, uplo, n, k, get_cste_float(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb, get_cste_float(beta),  get_ptr(c), ldc); break;
                    case dsyr2k:	cblas_dsyr2k(order, trans, uplo, n, k, get_cste_double(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb, get_cste_double(beta),  get_ptr(c), ldc); break;
                    case csyr2k:	cblas_csyr2k(order, trans, uplo, n, k,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,  get_cste_ptr(beta),  get_ptr(c), ldc); break;
                    case zsyr2k:	cblas_zsyr2k(order, trans, uplo, n, k,  get_cste_ptr(alpha),  get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,  get_cste_ptr(beta),  get_ptr(c), ldc); break;

                    default: error = ERROR_NOT_FOUND; break;
                }
                
            }
        break;}

//INSERT_HERE

        default:
            error = ERROR_NO_BLAS;
            debug_write("Error: blas %s of hash %lu does not exist.\n", name, hash(name));
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
    atomN           = enif_make_atom(env, "n");
    atomT           = enif_make_atom(env, "t");
    atomC           = enif_make_atom(env, "c");
    atomUpper       = enif_make_atom(env, "blasUpper");
    atomU           = enif_make_atom(env, "u");
    atomLower       = enif_make_atom(env, "blasLower");
    atomL           = enif_make_atom(env, "l");
    atomNonUnit     = enif_make_atom(env, "blasNonUnit");
    atomUnit        = enif_make_atom(env, "blasUnit");
    atomLeft        = enif_make_atom(env, "blasLeft");
    atomRight       = enif_make_atom(env, "blasRight");
    atomR           = enif_make_atom(env, "r");

    return 0;
}

ErlNifFunc nif_funcs[] = { 
    {"new_nif", 1, new},
    {"copy_nif", 2, copy},
    {"bin_nif", 2, to_binary},
    {"hash", 1, hash_nif},

    {"dirty_unwrapper", 2, unwrapper, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"clean_unwrapper", 2, unwrapper, 0}
};


ERL_NIF_INIT(blas, nif_funcs, load, NULL, NULL, NULL)