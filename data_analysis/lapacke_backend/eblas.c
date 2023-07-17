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

//INSERT_HERE

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