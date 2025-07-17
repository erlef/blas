#define STATIC_ERLANG_NIF 1

#include "erl_nif.h"
#include "string.h"
#include <complex.h>
#include "string.h"

#ifndef EWB_INCLUDED
#define EWB_INCLUDED
#include <cblas.h>
#include <lapacke.h>

// Types translator
ERL_NIF_TERM atomRowMajor, atomColMajor, atomNoTrans, atomTrans, atomConjTrans, atomUpper,atomLower, atomNonUnit, atomUnit, atomLeft, atomRight, atomN, atomT, atomC, atomU, atomL, atomR;

typedef enum types {e_int, e_uint, e_char, e_float, e_double, e_ptr, e_cste_ptr, e_float_complex, e_double_complex, e_layout, e_transpose, e_uplo, e_diag, e_side, e_end} etypes;
int translate(ErlNifEnv* env, const ERL_NIF_TERM* terms, const etypes* format, ...);



// C binary definition
// --------------------------------------------

typedef struct{
    unsigned int size;
    unsigned int offset;
    unsigned char* ptr;
} c_binary;

inline void* get_ptr(c_binary cb){return (void*) cb.ptr + cb.offset;}
int get_c_binary(ErlNifEnv* env, const ERL_NIF_TERM term, c_binary* result);
int in_bounds(int elem_size, int n_elem, int inc, c_binary b);

inline int leading_dim(int order, int trans, int n_rows, int n_cols){
    
    if(order == CblasRowMajor){
        return trans==CblasNoTrans? n_cols:n_rows;
    }
    
    if(order == CblasColMajor){
        return trans==CblasNoTrans? n_rows:n_cols;
    }

    return -1;
}

typedef struct{
    unsigned int size;
    unsigned int offset;
    const unsigned char* ptr;
    double tmp;
    etypes type;
} cste_c_binary;

inline const void* get_cste_ptr(cste_c_binary cb){return (void*) cb.ptr + cb.offset;}
float get_cste_float(cste_c_binary cb);
double get_cste_double(cste_c_binary cb);
int get_cste_binary(ErlNifEnv* env, const ERL_NIF_TERM term, cste_c_binary* result);
int in_cste_bounds(int elem_size, int n_elem, int inc, cste_c_binary b);

void set_cste_c_binary(cste_c_binary* ccb, etypes type, unsigned char* ptr);
ERL_NIF_TERM cste_c_binary_to_term(ErlNifEnv* env, cste_c_binary ccb);


// Private stuff
int debug_write(const char* fmt, ...);
ErlNifResourceType *c_binary_resource;
int load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info);
int upgrade(ErlNifEnv* caller_env, void** priv_data, void** old_priv_data, ERL_NIF_TERM load_info);
int unload(ErlNifEnv* caller_env, void* priv_data);

ERL_NIF_TERM new(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv);
ERL_NIF_TERM copy(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv);
ERL_NIF_TERM to_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv);



// Blas wrapper
// --------------------------------------------

unsigned long hash(char *str);
int load_ebw(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info);
ERL_NIF_TERM unwrapper(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv);
ERL_NIF_TERM blas_hash_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv);



#endif

typedef enum sizes {s_bytes=4, d_bytes=8, c_bytes=8, z_bytes=16, no_bytes=0} size_in_bytes;

typedef enum BLAS_NAMES {
    saxpy=210727551034,
    daxpy=210709762219,
    caxpy=210708576298,
    zaxpy=210735852481,
    scopy=210727613107,
    dcopy=210709824292,
    ccopy=210708638371,
    zcopy=210735914554,
    sswap=210728196307,
    dswap=210710407492,
    cswap=210709221571,
    zswap=210736497754,
    sscal=210728174523,
    dscal=210710385708,
    cscal=210709199787,
    csscal=6953404169886,
    zscal=210736475970,
    zdscal=6954286495110,
    sdot=6385686335,
    ddot=6385147280,
    cdotu=210708674436,
    zdotu=210735950619,
    cdotc=210708674418,
    zdotc=210735950601,
    dsdot=210710387267,
    sdsdot=6954012548918,
    snrm2=210728011511,
    dnrm2=210710222696,
    scnrm2=6954011198426,
    dznrm2=6953451443714,
    sasum=210727545742,
    dasum=210709756927,
    scasum=6954010732657,
    dzasum=6953450977945,
    isamax=6953638346280,
    idamax=6953620557465,
    icamax=6953619371544,
    izamax=6953646647727,
    srot=6385701581,
    drot=6385162526,
    csrot=210709216592,
    zdrot=210735953720,
    srotg=210728152276,
    drotg=210710363461,
    crotg=210709177540,
    zrotg=210736453723,
    srotmg=6954029025409,
    drotmg=6953441994514,
    srotm=210728152282,
    drotm=210710363467, 
    isamin=6953638346534,
    idamin=6953620557719,
    icamin=6953619371798,
    izamin=6953646647981,
    ismax=210716326215,
    idmax=210715787160,
    icmax=210715751223,
    izmax=210716577774,
    ismin=210716326469,
    idmin=210715787414,
    icmin=210715751477,
    izmin=210716578028,
    sgemv=210727745863,
    dgemv=210709957048,
    cgemv=210708771127,
    zgemv=210736047310,
    sgbmv=210727742596,
    dgbmv=210709953781,
    cgbmv=210708767860,
    zgbmv=210736044043,
    ssbmv=210728173840,
    dsbmv=210710385025,
    sger=6385689270,
    dger=6385150215,
    strmv=210728227201,
    dtrmv=210710438386,
    ctrmv=210709252465,
    ztrmv=210736528648,
    strsv=210728227399,
    dtrsv=210710438584,
    ctrsv=210709252663,
    ztrsv=210736528846,
    strsm=210728227390,
    dtrsm=210710438575,
    ctrsm=210709252654,
    ztrsm=210736528837,
    cgeru=210708771291,
    cgerc=210708771273,
    zgeru=210736047474,
    zgerc=210736047456,
    sgemm=210727745854,
    dgemm=210709957039,
    cgemm=210708771118,
    cgemm3m=229461851749294,
    zgemm=210736047301,
    zgemm3m=229491555512581,
    stbmv=210728209777,
    dtbmv=210710420962,
    ctbmv=210709235041,
    ztbmv=210736511224,
    stbsv=210728209975,
    dtbsv=210710421160,
    ctbsv=210709235239,
    ztbsv=210736511422,
    stpmv=210728225023,
    dtpmv=210710436208,
    ctpmv=210709250287,
    ztpmv=210736526470,
    stpsv=210728225221,
    dtpsv=210710436406,
    ctpsv=210709250485,
    ztpsv=210736526668,
    ssymv=210728198887, 
    dsymv=210710410072,
    chemv=210708807064,
    zhemv=210736083247,
    sspmv=210728189086,
    dspmv=210710400271,
    sspr=6385702701,
    dspr=6385163646,
    chpr=6385115730,
    zhpr=6385942281,
    sspr2=210728189183,
    dspr2=210710400368,
    chpr2=210708819140,
    zhpr2=210736095323,
    chbmv=210708803797,
    zhbmv=210736079980,
    chpmv=210708819043,
    zhpmv=210736095226,
    cher=6385115367,
    zher=6385941918,
    chemm=210708807055,
    zhemm=210736083238,
    cherk=210708807218,
    zherk=210736083401,
    cher2k=6953390636420,
    zher2k=6954290750459,
    ssymm=210728198878,
    dsymm=210710410063,
    csymm=210709224142,
    zsymm=210736500325,
    ssyrk=210728199041,
    dsyrk=210710410226,
    csyrk=210709224305,
    zsyrk=210736500488,
    ssyr2k=6954030566579,
    dsyr2k=6953443535684,
    csyr2k=6953404400291,
    zsyr2k=6954304514330,
    ssum=6385702861,
    dsum=6385163806,
    dzsum=210710655352,
    scsum=210727617616,
    cher2=210708807161,
    zher2=210736083344,
    strmm=210728227192,
    dtrmm=210710438377,
    ctrmm=210709252456,
    ztrmm=210736528639,
    ssyr=6385702998,
    dsyr=6385163943,
    ssyr2=210728198984,
    dsyr2=210710410169,
   sbdsdc = 6954009653976,
   dbdsdc = 6953422623081,
   sbdsqr = 6954009654420,
   dbdsqr = 6953422623525,
   cbdsqr = 6953383488132,
   zbdsqr = 6954283602171,
   sdisna = 6954012205831,
   ddisna = 6953425174936,
   sgbbrd = 6954015493657,
   dgbbrd = 6953428462762,
   cgbbrd = 6953389327369,
   zgbbrd = 6954289441408,
   sgbcon = 6954015494657,
   dgbcon = 6953428463762,
   cgbcon = 6953389328369,
   zgbcon = 6954289442408,
   sgbequ = 6954015496908,
   dgbequ = 6953428466013,
   cgbequ = 6953389330620,
   zgbequ = 6954289444659,
   sgbequb = 229482511398062,
   dgbequb = 229463139378527,
   cgbequb = 229461847910558,
   zgbequb = 229491551673845,
   sgbrfs = 6954015510700,
   dgbrfs = 6953428479805,
   cgbrfs = 6953389344412,
   zgbrfs = 6954289458451,
   sgbsv = 210727742794,
   dgbsv = 210709953979,
   cgbsv = 210708768058,
   zgbsv = 210736044241,
   sgbtrf = 6954015513261,
   dgbtrf = 6953428482366,
   cgbtrf = 6953389346973,
   zgbtrf = 6954289461012,
   sgbtrs = 6954015513274,
   dgbtrs = 6953428482379,
   cgbtrs = 6953389346986,
   zgbtrs = 6954289461025,
   sgebak = 6954015600914,
   dgebak = 6953428570019,
   cgebak = 6953389434626,
   zgebak = 6954289548665,
   sgebal = 6954015600915,
   dgebal = 6953428570020,
   cgebal = 6953389434627,
   zgebal = 6954289548666,
   sgebrd = 6954015601468,
   dgebrd = 6953428570573,
   cgebrd = 6953389435180,
   zgebrd = 6954289549219,
   sgecon = 6954015602468,
   dgecon = 6953428571573,
   cgecon = 6953389436180,
   zgecon = 6954289550219,
   sgeequ = 6954015604719,
   dgeequ = 6953428573824,
   cgeequ = 6953389438431,
   zgeequ = 6954289552470,
   sgeequb = 229482514955825,
   dgeequb = 229463142936290,
   cgeequb = 229461851468321,
   zgeequb = 229491555231608,
   sgeev = 210727745599,
   dgeev = 210709956784,
   cgeev = 210708770863,
   zgeev = 210736047046,
   sgeevx = 6954015604887,
   dgeevx = 6953428573992,
   cgeevx = 6953389438599,
   zgeevx = 6954289552638,
   sgehrd = 6954015608002,
   dgehrd = 6953428577107,
   cgehrd = 6953389441714,
   zgehrd = 6954289555753,
   sgejsv = 6954015610231,
   dgejsv = 6953428579336,
   sgelqf = 6954015612327,
   dgelqf = 6953428581432,
   cgelqf = 6953389446039,
   zgelqf = 6954289560078,
   sgels = 210727745827,
   dgels = 210709957012,
   cgels = 210708771091,
   zgels = 210736047274,
   sgelsd = 6954015612391,
   dgelsd = 6953428581496,
   cgelsd = 6953389446103,
   zgelsd = 6954289560142,
   sgelss = 6954015612406,
   dgelss = 6953428581511,
   cgelss = 6953389446118,
   zgelss = 6954289560157,
   sgelsy = 6954015612412,
   dgelsy = 6953428581517,
   cgelsy = 6953389446124,
   zgelsy = 6954289560163,
   sgeqlf = 6954015617607,
   dgeqlf = 6953428586712,
   cgeqlf = 6953389451319,
   zgeqlf = 6954289565358,
   sgeqp3 = 6954015617688,
   dgeqp3 = 6953428586793,
   cgeqp3 = 6953389451400,
   zgeqp3 = 6954289565439,
   sgeqpf = 6954015617739,
   dgeqpf = 6953428586844,
   cgeqpf = 6953389451451,
   zgeqpf = 6954289565490,
   sgeqrf = 6954015617805,
   dgeqrf = 6953428586910,
   cgeqrf = 6953389451517,
   zgeqrf = 6954289565556,
   sgeqrfp = 229482515387677,
   dgeqrfp = 229463143368142,
   cgeqrfp = 229461851900173,
   zgeqrfp = 229491555663460,
   sgerfs = 6954015618511,
   dgerfs = 6953428587616,
   cgerfs = 6953389452223,
   zgerfs = 6954289566262,
   sgerqf = 6954015618861,
   dgerqf = 6953428587966,
   cgerqf = 6953389452573,
   zgerqf = 6954289566612,
   sgesdd = 6954015619519,
   dgesdd = 6953428588624,
   cgesdd = 6953389453231,
   zgesdd = 6954289567270,
   sgesv = 210727746061,
   dgesv = 210709957246,
   cgesv = 210708771325,
   zgesv = 210736047508,
   sgesvd = 6954015620113,
   dgesvd = 6953428589218,
   cgesvd = 6953389453825,
   zgesvd = 6954289567864,
   sgesvj = 6954015620119,
   dgesvj = 6953428589224,
   sgetrf = 6954015621072,
   dgetrf = 6953428590177,
   cgetrf = 6953389454784,
   zgetrf = 6954289568823,
   sgetri = 6954015621075,
   dgetri = 6953428590180,
   cgetri = 6953389454787,
   zgetri = 6954289568826,
   sgetrs = 6954015621085,
   dgetrs = 6953428590190,
   cgetrs = 6953389454797,
   zgetrs = 6954289568836,
   sggbak = 6954015672788,
   dggbak = 6953428641893,
   cggbak = 6953389506500,
   zggbak = 6954289620539,
   sggbal = 6954015672789,
   dggbal = 6953428641894,
   cggbal = 6953389506501,
   zggbal = 6954289620540,
   sggev = 210727747777,
   dggev = 210709958962,
   cggev = 210708773041,
   zggev = 210736049224,
   sggevx = 6954015676761,
   dggevx = 6953428645866,
   cggevx = 6953389510473,
   zggevx = 6954289624512,
   sggglm = 6954015678598,
   dggglm = 6953428647703,
   cggglm = 6953389512310,
   zggglm = 6954289626349,
   sgghrd = 6954015679876,
   dgghrd = 6953428648981,
   cgghrd = 6953389513588,
   zgghrd = 6954289627627,
   sgglse = 6954015684266,
   dgglse = 6953428653371,
   cgglse = 6953389517978,
   zgglse = 6954289632017,
   sggqrf = 6954015689679,
   dggqrf = 6953428658784,
   cggqrf = 6953389523391,
   zggqrf = 6954289637430,
   sggrqf = 6954015690735,
   dggrqf = 6953428659840,
   cggrqf = 6953389524447,
   zggrqf = 6954289638486,
   sggsvd = 6954015691987,
   dggsvd = 6953428661092,
   cggsvd = 6953389525699,
   zggsvd = 6954289639738,
   sggsvp = 6954015691999,
   dggsvp = 6953428661104,
   cggsvp = 6953389525711,
   zggsvp = 6954289639750,
   sgtcon = 6954016141523,
   dgtcon = 6953429110628,
   cgtcon = 6953389975235,
   zgtcon = 6954290089274,
   sgtrfs = 6954016157566,
   dgtrfs = 6953429126671,
   cgtrfs = 6953389991278,
   zgtrfs = 6954290105317,
   sgtsv = 210727762396,
   dgtsv = 210709973581,
   cgtsv = 210708787660,
   zgtsv = 210736063843,
   sgtsvx = 6954016159188,
   dgtsvx = 6953429128293,
   cgtsvx = 6953389992900,
   zgtsvx = 6954290106939,
   sgttrf = 6954016160127,
   dgttrf = 6953429129232,
   cgttrf = 6953389993839,
   zgttrf = 6954290107878,
   sgttrs = 6954016160140,
   dgttrs = 6953429129245,
   cgttrs = 6953389993852,
   zgttrs = 6954290107891,
   chbev = 210708803533,
   zhbev = 210736079716,
   chbevd = 6953390516689,
   zhbevd = 6954290630728,
   chbevx = 6953390516709,
   zhbevx = 6954290630748,
   chbgst = 6953390518784,
   zhbgst = 6954290632823,
   chbgv = 210708803599,
   zhbgv = 210736079782,
   chbgvd = 6953390518867,
   zhbgvd = 6954290632906,
   chbgvx = 6953390518887,
   zhbgvx = 6954290632926,
   chbtrd = 6953390532892,
   zhbtrd = 6954290646931,
   checon = 6953390622101,
   zhecon = 6954290736140,
   cheequb = 229461890603714,
   zheequb = 229491594367001,
   cheev = 210708806800,
   zheev = 210736082983,
   cheevd = 6953390624500,
   zheevd = 6954290738539,
   cheevr = 6953390624514,
   zheevr = 6954290738553,
   cheevx = 6953390624520,
   zheevx = 6954290738559,
   chegst = 6953390626595,
   zhegst = 6954290740634,
   chegv = 210708806866,
   zhegv = 210736083049,
   chegvd = 6953390626678,
   zhegvd = 6954290740717,
   chegvx = 6953390626698,
   zhegvx = 6954290740737,
   cherfs = 6953390638144,
   zherfs = 6954290752183,
   chesv = 210708807262,
   zhesv = 210736083445,
   chesvx = 6953390639766,
   zhesvx = 6954290753805,
   chetrd = 6953390640703,
   zhetrd = 6954290754742,
   chetrf = 6953390640705,
   zhetrf = 6954290754744,
   chetri = 6953390640708,
   zhetri = 6954290754747,
   chetrs = 6953390640718,
   zhetrs = 6954290754757,
   chfrk = 210708808307,
   zhfrk = 210736084490,
   shgeqz = 6954016862519,
   dhgeqz = 6953429831624,
   chgeqz = 6953390696231,
   zhgeqz = 6954290810270,
   chpcon = 6953391017408,
   zhpcon = 6954291131447,
   chpev = 210708818779,
   zhpev = 210736094962,
   chpevd = 6953391019807,
   zhpevd = 6954291133846,
   chpevx = 6953391019827,
   zhpevx = 6954291133866,
   chpgst = 6953391021902,
   zhpgst = 6954291135941,
   chpgv = 210708818845,
   zhpgv = 210736095028,
   chpgvd = 6953391021985,
   zhpgvd = 6954291136024,
   chpgvx = 6953391022005,
   zhpgvx = 6954291136044,
   chprfs = 6953391033451,
   zhprfs = 6954291147490,
   chpsv = 210708819241,
   zhpsv = 210736095424,
   chpsvx = 6953391035073,
   zhpsvx = 6954291149112,
   chptrd = 6953391036010,
   zhptrd = 6954291150049,
   chptrf = 6953391036012,
   zhptrf = 6954291150051,
   chptri = 6953391036015,
   zhptri = 6954291150054,
   chptrs = 6953391036025,
   zhptrs = 6954291150064,
   shsein = 6954017293487,
   dhsein = 6953430262592,
   chsein = 6953391127199,
   zhsein = 6954291241238,
   shseqr = 6954017293755,
   dhseqr = 6953430262860,
   chseqr = 6953391127467,
   zhseqr = 6954291241506,
   sopgtr = 6954025489668,
   dopgtr = 6953438458773,
   sopmtr = 6954025496202,
   dopmtr = 6953438465307,
   sorgbr = 6954025560948,
   dorgbr = 6953438530053,
   sorghr = 6954025561146,
   dorghr = 6953438530251,
   sorglq = 6954025561277,
   dorglq = 6953438530382,
   sorgql = 6954025561437,
   dorgql = 6953438530542,
   sorgqr = 6954025561443,
   dorgqr = 6953438530548,
   sorgrq = 6954025561475,
   dorgrq = 6953438530580,
   sorgtr = 6954025561542,
   dorgtr = 6953438530647,
   sormbr = 6954025567482,
   dormbr = 6953438536587,
   sormhr = 6954025567680,
   dormhr = 6953438536785,
   sormlq = 6954025567811,
   dormlq = 6953438536916,
   sormql = 6954025567971,
   dormql = 6953438537076,
   sormqr = 6954025567977,
   dormqr = 6953438537082,
   sormrq = 6954025568009,
   dormrq = 6953438537114,
   sormrz = 6954025568018,
   dormrz = 6953438537123,
   sormtr = 6954025568076,
   dormtr = 6953438537181,
   spbcon = 6954026167946,
   dpbcon = 6953439137051,
   cpbcon = 6953400001658,
   zpbcon = 6954300115697,
   spbequ = 6954026170197,
   dpbequ = 6953439139302,
   cpbequ = 6953400003909,
   zpbequ = 6954300117948,
   spbrfs = 6954026183989,
   dpbrfs = 6953439153094,
   cpbrfs = 6953400017701,
   zpbrfs = 6954300131740,
   spbstf = 6954026185527,
   dpbstf = 6953439154632,
   cpbstf = 6953400019239,
   zpbstf = 6954300133278,
   spbsv = 210728066227,
   dpbsv = 210710277412,
   cpbsv = 210709091491,
   zpbsv = 210736367674,
   spbtrf = 6954026186550,
   dpbtrf = 6953439155655,
   cpbtrf = 6953400020262,
   zpbtrf = 6954300134301,
   spbtrs = 6954026186563,
   dpbtrs = 6953439155668,
   cpbtrs = 6953400020275,
   zpbtrs = 6954300134314,
   spftrf = 6954026330298,
   dpftrf = 6953439299403,
   cpftrf = 6953400164010,
   zpftrf = 6954300278049,
   spftri = 6954026330301,
   dpftri = 6953439299406,
   cpftri = 6953400164013,
   zpftri = 6954300278052,
   spftrs = 6954026330311,
   dpftrs = 6953439299416,
   cpftrs = 6953400164023,
   zpftrs = 6954300278062,
   spocon = 6954026635127,
   dpocon = 6953439604232,
   cpocon = 6953400468839,
   zpocon = 6954300582878,
   spoequ = 6954026637378,
   dpoequ = 6953439606483,
   cpoequ = 6953400471090,
   zpoequ = 6954300585129,
   spoequb = 229482879033572,
   dpoequb = 229463507014037,
   cpoequb = 229462215546068,
   zpoequb = 229491919309355,
   sporfs = 6954026651170,
   dporfs = 6953439620275,
   cporfs = 6953400484882,
   zporfs = 6954300598921,
   sposv = 210728080384,
   dposv = 210710291569,
   cposv = 210709105648,
   zposv = 210736381831,
   spotrf = 6954026653731,
   dpotrf = 6953439622836,
   cpotrf = 6953400487443,
   zpotrf = 6954300601482,
   spotri = 6954026653734,
   dpotri = 6953439622839,
   cpotri = 6953400487446,
   zpotri = 6954300601485,
   spotrs = 6954026653744,
   dpotrs = 6953439622849,
   cpotrs = 6953400487456,
   zpotrs = 6954300601495,
   sppcon = 6954026671064,
   dppcon = 6953439640169,
   cppcon = 6953400504776,
   zppcon = 6954300618815,
   sppequ = 6954026673315,
   dppequ = 6953439642420,
   cppequ = 6953400507027,
   zppequ = 6954300621066,
   spprfs = 6954026687107,
   dpprfs = 6953439656212,
   cpprfs = 6953400520819,
   zpprfs = 6954300634858,
   sppsv = 210728081473,
   dppsv = 210710292658,
   cppsv = 210709106737,
   zppsv = 210736382920,
   spptrf = 6954026689668,
   dpptrf = 6953439658773,
   cpptrf = 6953400523380,
   zpptrf = 6954300637419,
   spptri = 6954026689671,
   dpptri = 6953439658776,
   cpptri = 6953400523383,
   zpptri = 6954300637422,
   spptrs = 6954026689681,
   dpptrs = 6953439658786,
   cpptrs = 6953400523393,
   zpptrs = 6954300637432,
   spstrf = 6954026797479,
   dpstrf = 6953439766584,
   cpstrf = 6953400631191,
   zpstrf = 6954300745230,
   sptcon = 6954026814812,
   dptcon = 6953439783917,
   cptcon = 6953400648524,
   zptcon = 6954300762563,
   spteqr = 6954026817060,
   dpteqr = 6953439786165,
   cpteqr = 6953400650772,
   zpteqr = 6954300764811,
   sptrfs = 6954026830855,
   dptrfs = 6953439799960,
   cptrfs = 6953400664567,
   zptrfs = 6954300778606,
   sptsv = 210728085829,
   dptsv = 210710297014,
   cptsv = 210709111093,
   zptsv = 210736387276,
   sptsvx = 6954026832477,
   dptsvx = 6953439801582,
   cptsvx = 6953400666189,
   zptsvx = 6954300780228,
   spttrf = 6954026833416,
   dpttrf = 6953439802521,
   cpttrf = 6953400667128,
   zpttrf = 6954300781167,
   spttrs = 6954026833429,
   dpttrs = 6953439802534,
   cpttrs = 6953400667141,
   zpttrs = 6954300781180,
   ssbev = 210728173576,
   dsbev = 210710384761,
   ssbevd = 6954029728108,
   dsbevd = 6953442697213,
   ssbevx = 6954029728128,
   dsbevx = 6953442697233,
   ssbgst = 6954029730203,
   dsbgst = 6953442699308,
   ssbgv = 210728173642,
   dsbgv = 210710384827,
   ssbgvd = 6954029730286,
   dsbgvd = 6953442699391,
   ssbgvx = 6954029730306,
   dsbgvx = 6953442699411,
   ssbtrd = 6954029744311,
   dsbtrd = 6953442713416,
   ssfrk = 210728178350,
   dsfrk = 210710389535,
   sspcon = 6954030228827,
   dspcon = 6953443197932,
   cspcon = 6953404062539,
   zspcon = 6954304176578,
   sspev = 210728188822,
   dspev = 210710400007,
   sspevd = 6954030231226,
   dspevd = 6953443200331,
   sspevx = 6954030231246,
   dspevx = 6953443200351,
   sspgst = 6954030233321,
   dspgst = 6953443202426,
   sspgv = 210728188888,
   dspgv = 210710400073,
   sspgvd = 6954030233404,
   dspgvd = 6953443202509,
   sspgvx = 6954030233424,
   dspgvx = 6953443202529,
   ssprfs = 6954030244870,
   dsprfs = 6953443213975,
   csprfs = 6953404078582,
   zsprfs = 6954304192621,
   sspsv = 210728189284,
   dspsv = 210710400469,
   cspsv = 210709214548,
   zspsv = 210736490731,
   sspsvx = 6954030246492,
   dspsvx = 6953443215597,
   cspsvx = 6953404080204,
   zspsvx = 6954304194243,
   ssptrd = 6954030247429,
   dsptrd = 6953443216534,
   ssptrf = 6954030247431,
   dsptrf = 6953443216536,
   csptrf = 6953404081143,
   zsptrf = 6954304195182,
   ssptri = 6954030247434,
   dsptri = 6953443216539,
   csptri = 6953404081146,
   zsptri = 6954304195185,
   ssptrs = 6954030247444,
   dsptrs = 6953443216549,
   csptrs = 6953404081156,
   zsptrs = 6954304195195,
   sstebz = 6954030374336,
   dstebz = 6953443343441,
   sstedc = 6954030374379,
   dstedc = 6953443343484,
   cstedc = 6953404208091,
   zstedc = 6954304322130,
   sstegr = 6954030374493,
   dstegr = 6953443343598,
   cstegr = 6953404208205,
   zstegr = 6954304322244,
   sstein = 6954030374555,
   dstein = 6953443343660,
   cstein = 6953404208267,
   zstein = 6954304322306,
   sstemr = 6954030374691,
   dstemr = 6953443343796,
   cstemr = 6953404208403,
   zstemr = 6954304322442,
   ssteqr = 6954030374823,
   dsteqr = 6953443343928,
   csteqr = 6953404208535,
   zsteqr = 6954304322574,
   ssterf = 6954030374844,
   dsterf = 6953443343949,
   sstev = 210728193178,
   dstev = 210710404363,
   sstevd = 6954030374974,
   dstevd = 6953443344079,
   sstevr = 6954030374988,
   dstevr = 6953443344093,
   sstevx = 6954030374994,
   dstevx = 6953443344099,
   ssycon = 6954030552260,
   dsycon = 6953443521365,
   csycon = 6953404385972,
   zsycon = 6954304500011,
   ssyequb = 229483008298961,
   dsyequb = 229463636279426,
   csyequb = 229462344811457,
   zsyequb = 229492048574744,
   ssyev = 210728198623,
   dsyev = 210710409808,
   ssyevd = 6954030554659,
   dsyevd = 6953443523764,
   ssyevr = 6954030554673,
   dsyevr = 6953443523778,
   ssyevx = 6954030554679,
   dsyevx = 6953443523784,
   ssygst = 6954030556754,
   dsygst = 6953443525859,
   ssygv = 210728198689,
   dsygv = 210710409874,
   ssygvd = 6954030556837,
   dsygvd = 6953443525942,
   ssygvx = 6954030556857,
   dsygvx = 6953443525962,
   ssyrfs = 6954030568303,
   dsyrfs = 6953443537408,
   csyrfs = 6953404402015,
   zsyrfs = 6954304516054,
   ssysv = 210728199085,
   dsysv = 210710410270,
   csysv = 210709224349,
   zsysv = 210736500532,
   ssysvx = 6954030569925,
   dsysvx = 6953443539030,
   csysvx = 6953404403637,
   zsysvx = 6954304517676,
   ssytrd = 6954030570862,
   dsytrd = 6953443539967,
   ssytrf = 6954030570864,
   dsytrf = 6953443539969,
   csytrf = 6953404404576,
   zsytrf = 6954304518615,
   ssytri = 6954030570867,
   dsytri = 6953443539972,
   csytri = 6953404404579,
   zsytri = 6954304518618,
   ssytrs = 6954030570877,
   dsytrs = 6953443539982,
   csytrs = 6953404404589,
   zsytrs = 6954304518628,
   stbcon = 6954030911630,
   dtbcon = 6953443880735,
   ctbcon = 6953404745342,
   ztbcon = 6954304859381,
   stbrfs = 6954030927673,
   dtbrfs = 6953443896778,
   ctbrfs = 6953404761385,
   ztbrfs = 6954304875424,
   stbtrs = 6954030930247,
   dtbtrs = 6953443899352,
   ctbtrs = 6953404763959,
   ztbtrs = 6954304877998,
   stfsm = 210728214322,
   dtfsm = 210710425507,
   ctfsm = 210709239586,
   ztfsm = 210736515769,
   stftri = 6954031073985,
   dtftri = 6953444043090,
   ctftri = 6953404907697,
   ztftri = 6954305021736,
   stfttp = 6954031074058,
   dtfttp = 6953444043163,
   ctfttp = 6953404907770,
   ztfttp = 6954305021809,
   stfttr = 6954031074060,
   dtfttr = 6953444043165,
   ctfttr = 6953404907772,
   ztfttr = 6954305021811,
   stgevc = 6954031093713,
   dtgevc = 6953444062818,
   ctgevc = 6953404927425,
   ztgevc = 6954305041464,
   stgexc = 6954031093779,
   dtgexc = 6953444062884,
   ctgexc = 6953404927491,
   ztgexc = 6954305041530,
   stgsen = 6954031108409,
   dtgsen = 6953444077514,
   ctgsen = 6953404942121,
   ztgsen = 6954305056160,
   stgsja = 6954031108561,
   dtgsja = 6953444077666,
   ctgsja = 6953404942273,
   ztgsja = 6954305056312,
   stgsna = 6954031108693,
   dtgsna = 6953444077798,
   ctgsna = 6953404942405,
   ztgsna = 6954305056444,
   stgsyl = 6954031109067,
   dtgsyl = 6953444078172,
   ctgsyl = 6953404942779,
   ztgsyl = 6954305056818,
   stpcon = 6954031414748,
   dtpcon = 6953444383853,
   ctpcon = 6953405248460,
   ztpcon = 6954305362499,
   stprfs = 6954031430791,
   dtprfs = 6953444399896,
   ctprfs = 6953405264503,
   ztprfs = 6954305378542,
   stptri = 6954031433355,
   dtptri = 6953444402460,
   ctptri = 6953405267067,
   ztptri = 6954305381106,
   stptrs = 6954031433365,
   dtptrs = 6953444402470,
   ctptrs = 6953405267077,
   ztptrs = 6954305381116,
   stpttf = 6954031433418,
   dtpttf = 6953444402523,
   ctpttf = 6953405267130,
   ztpttf = 6954305381169,
   stpttr = 6954031433430,
   dtpttr = 6953444402535,
   ctpttr = 6953405267142,
   ztpttr = 6954305381181,
   strcon = 6954031486622,
   dtrcon = 6953444455727,
   ctrcon = 6953405320334,
   ztrcon = 6954305434373,
   strevc = 6954031489020,
   dtrevc = 6953444458125,
   ctrevc = 6953405322732,
   ztrevc = 6954305436771,
   strexc = 6954031489086,
   dtrexc = 6953444458191,
   ctrexc = 6953405322798,
   ztrexc = 6954305436837,
   strrfs = 6954031502665,
   dtrrfs = 6953444471770,
   ctrrfs = 6953405336377,
   ztrrfs = 6954305450416,
   strsen = 6954031503716,
   dtrsen = 6953444472821,
   ctrsen = 6953405337428,
   ztrsen = 6954305451467,
   strsna = 6954031504000,
   dtrsna = 6953444473105,
   ctrsna = 6953405337712,
   ztrsna = 6954305451751,
   strsyl = 6954031504374,
   dtrsyl = 6953444473479,
   ctrsyl = 6953405338086,
   ztrsyl = 6954305452125,
   strtri = 6954031505229,
   dtrtri = 6953444474334,
   ctrtri = 6953405338941,
   ztrtri = 6954305452980,
   strtrs = 6954031505239,
   dtrtrs = 6953444474344,
   ctrtrs = 6953405338951,
   ztrtrs = 6954305452990,
   strttf = 6954031505292,
   dtrttf = 6953444474397,
   ctrttf = 6953405339004,
   ztrttf = 6954305453043,
   strttp = 6954031505302,
   dtrttp = 6953444474407,
   ctrttp = 6953405339014,
   ztrttp = 6954305453053,
   stzrzf = 6954031790808,
   dtzrzf = 6953444759913,
   ctzrzf = 6953405624520,
   ztzrzf = 6954305738559,
   cungbr = 6953406366438,
   zungbr = 6954306480477,
   cunghr = 6953406366636,
   zunghr = 6954306480675,
   cunglq = 6953406366767,
   zunglq = 6954306480806,
   cungql = 6953406366927,
   zungql = 6954306480966,
   cungqr = 6953406366933,
   zungqr = 6954306480972,
   cungrq = 6953406366965,
   zungrq = 6954306481004,
   cungtr = 6953406367032,
   zungtr = 6954306481071,
   cunmbr = 6953406372972,
   zunmbr = 6954306487011,
   cunmhr = 6953406373170,
   zunmhr = 6954306487209,
   cunmlq = 6953406373301,
   zunmlq = 6954306487340,
   cunmql = 6953406373461,
   zunmql = 6954306487500,
   cunmqr = 6953406373467,
   zunmqr = 6954306487506,
   cunmrq = 6953406373499,
   zunmrq = 6954306487538,
   cunmrz = 6953406373508,
   zunmrz = 6954306487547,
   cunmtr = 6953406373566,
   zunmtr = 6954306487605,
   cupgtr = 6953406438906,
   zupgtr = 6954306552945,
   cupmtr = 6953406445440,
   zupmtr = 6954306559479,

    blas_name_end=0
} blas_names;


size_in_bytes pick_size(long unsigned hash);
size_in_bytes pick_size(long unsigned hash){
    size_in_bytes type;

     switch(hash){
        case saxpy: case scopy: case sswap: case sscal: case sdot: case srot: 
        case srotmg:case sgemv: case sger:  case sgbmv: case ssbmv:case srotg: 
        case snrm2: case sasum: case isamax:case isamin:case ismax:case ismin:
        case srotm: case sdsdot: case ssum:
        case strmv: case strmm: case strsv: case strsm:
        case sgemm: case sspmv: case sspr: case sspr2:
        case stpsv: case stpmv: case stbsv: case stbmv: case scasum: case scsum: case scnrm2:
        case ssymv: case ssymm: case ssyrk: case ssyr2k:
        case ssyr: case ssyr2: case dsdot:
            type = s_bytes;  
        break;   

        case daxpy: case dcopy: case dswap:  case dscal: case ddot: case drot:
        case drotg: case drotm: case drotmg: case dgemv: case dger: case dgbmv: 
        case dnrm2: case dasum: case idamax: case idamin:case idmax:case idmin: 
        case dsbmv: case dsum: 
        case dtrmv: case dtrsv: case dtrmm: case dtrsm:
        case dgemm: case dspmv: case dspr: case dspr2:
        case dtpsv: case dtpmv: case dtbsv: case dtbmv:
        case dsymv: case dsymm: case dsyrk:
        case dsyr2k:case dsyr: case dsyr2:
            type = d_bytes;  
        break;

        case caxpy: case ccopy: case cscal: case cdotu: case cgemv: case cgeru: 
        case icamax:case icamin:case icmax: case icmin: 
        case cgerc: case cgbmv: case cswap: case csscal: case cdotc:
        case ctrmv: case ctrsv: case ctrmm: case ctrsm: 
        case cgemm: case cgemm3m: case csrot:
        case ctpsv: case ctpmv: case ctbsv: case ctbmv:
        case crotg:case cher2:
        case chemv:case chpr:case chpr2:case chbmv:
        case chpmv:case cher:case chemm:case cherk:
        case cher2k:case csymm:case csyrk:case csyr2k:
            type = c_bytes;  
        break;

        case zaxpy: case zcopy: case zswap: case zscal: case zdotu: case zdotc: 
        case dznrm2:case dzasum:case izamax:case izamin:case izmax: case izmin: 
        case zrotg: case zgemv: case zgeru: case zgerc: case zgbmv: case dzsum:
        case ztrmv: case ztrsv: case ztrmm: case ztrsm: case zdscal: case zdrot:
        case zgemm: case zgemm3m: case ztbmv:
        case ztpsv: case ztbsv:
        case ztpmv:
        case zhemv:case zhpr:case zhpr2:
        case zhbmv:case zhpmv:case zher:
        case zhemm:case zherk:case zher2k:
        case zsymm:case zsyrk:case zsyr2k:
        case zher2:
            type = z_bytes;  
        break;
        
        default:
            type = no_bytes;
        break;
    }

    return type;
}


typedef enum errors {
    ERROR_SIGSEV    = 20,   // Array overflow
    ERROR_N_ARG     = 21,   // Invalid number of arguments
    ERROR_NOT_FOUND = 404,  // Switch case branch not solved.
    ERROR_NO_BLAS   = -1,   // Invalid blas name
    ERROR_NONE      = 0    // No error.
} errors;

inline errors test_n_arg(int narg, int expected){
    return narg == expected? ERROR_NONE:ERROR_N_ARG;
}

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

            case e_char: {
                char* c_dest = va_arg(valist, char*);
                char* buff = "0";
                enif_get_atom(env, terms[curr], buff, 2, ERL_NIF_LATIN1);
                c_dest[0] = buff[0];
            break;}
            
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
        case e_int:{     int    vali = *(int*)    ccb.ptr; result = enif_make_int(env, vali);    break;}
        case e_double:{  double vald = *(double*) ccb.ptr; result = enif_make_double(env, vald); break;}

        case e_float_complex:
        case e_double_complex:{
             ErlNifBinary bin;

            if(enif_alloc_binary(ccb.size, &bin)){
                memcpy(bin.data, ccb.ptr, ccb.size);
                if(!(result = enif_make_binary(env, &bin))){
                    enif_release_binary(&bin);
                    result = enif_make_badarg(env);
                }
            }
        break;}

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

            double f_result;
            double d_result;
            double ds_result;
            complex float c_result;
            complex double z_result;
            complex float cd_result;
            complex double zd_result;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_end}, &n, &x, &incx, &y, &incy))
                && !(error = in_cste_bounds(type, n, incx, x) ) && !(error = in_cste_bounds(type, n, incy, y))
            ){ 
                switch(hash_name){
                    case sdot:                   f_result  = cblas_sdot (n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &f_result);  break;
                    case ddot:                   d_result  = cblas_ddot (n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &d_result);  break;
                    case dsdot:                  ds_result = cblas_dsdot(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double, (unsigned char*) &ds_result); break;
                    //case cdotu: openblas_complex_float  c_result  = cblas_cdotu(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_float_complex,  (unsigned char*) &c_result);  break;
                    //case zdotu: openblas_complex_double z_result  = cblas_zdotu(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double_complex, (unsigned char*) &z_result);  break;
                    //case cdotc: openblas_complex_float  cd_result = cblas_cdotc(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_float_complex,  (unsigned char*) &cd_result); break;
                    //case zdotc: openblas_complex_double zd_result = cblas_zdotc(n, get_cste_ptr(x), incx, get_cste_ptr(y), incy); set_cste_c_binary(&dot_result, e_double_complex, (unsigned char*) &zd_result); break;
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
                    
                    //case dsum:  d_result  = cblas_dsum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    //case ssum:  d_result  = cblas_ssum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    //case scsum: d_result  = cblas_scsum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    //case dzsum: d_result  = cblas_dzsum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    
                    case dasum:  d_result  = cblas_dasum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case sasum:  d_result  = cblas_sasum (n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case scasum: d_result  = cblas_scasum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    case dzasum: d_result  = cblas_dzasum(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_double, (unsigned char*) &d_result);  break;
                    
                    case isamax: i_result  = cblas_isamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case idamax: i_result  = cblas_idamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case icamax: i_result  = cblas_icamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    case izamax: i_result  = cblas_izamax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    //case isamin: i_result  = cblas_isamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    //case idamin: i_result  = cblas_idamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    //case icamin: i_result  = cblas_icamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    //case izamin: i_result  = cblas_izamin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    //case ismax: i_result  = cblas_ismax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    //case idmax: i_result  = cblas_idmax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    //case icmax: i_result  = cblas_icmax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    //case izmax: i_result  = cblas_izmax(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

                    //case ismin: i_result  = cblas_ismin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    //case idmin: i_result  = cblas_idmin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    //case icmin: i_result  = cblas_icmin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;
                    //case izmin: i_result  = cblas_izmin(n, get_cste_ptr(x), incx); set_cste_c_binary(&u_result, e_int, (unsigned char*) &i_result);  break;

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
				//case cgemm3m: cblas_cgemm3m(order, transa, transb, m, n, k,  get_cste_ptr(alpha),     get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,    get_cste_ptr(beta),  get_ptr(c), ldc); break;
				case zgemm:	  cblas_zgemm(  order, transa, transb, m, n, k,  get_cste_ptr(alpha),     get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,    get_cste_ptr(beta),  get_ptr(c), ldc); break;
				//case zgemm3m: cblas_zgemm3m(order, transa, transb, m, n, k,  get_cste_ptr(alpha),     get_cste_ptr(a), lda,  get_cste_ptr(b), ldb,    get_cste_ptr(beta),  get_ptr(c), ldc); break;

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


        case sbdsdc: {
            int matrix_layout; char uplo; char compq; int n; c_binary d; c_binary e; c_binary u; int ldu; c_binary vt; int ldvt; c_binary q; c_binary iq;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &compq, &n, &d, &e, &u, &ldu, &vt, &ldvt, &q, &iq))
            ){
                LAPACKE_sbdsdc(matrix_layout, uplo, compq, n, get_ptr(d), get_ptr(e), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(q), get_ptr(iq));
            }
        break; }
        case dbdsdc: {
            int matrix_layout; char uplo; char compq; int n; c_binary d; c_binary e; c_binary u; int ldu; c_binary vt; int ldvt; c_binary q; c_binary iq;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &compq, &n, &d, &e, &u, &ldu, &vt, &ldvt, &q, &iq))
            ){
                LAPACKE_dbdsdc(matrix_layout, uplo, compq, n, get_ptr(d), get_ptr(e), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(q), get_ptr(iq));
            }
        break; }
        case sbdsqr: {
            int matrix_layout; char uplo; int n; int ncvt; int nru; int ncc; c_binary d; c_binary e; c_binary vt; int ldvt; c_binary u; int ldu; c_binary c; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc))
            ){
                LAPACKE_sbdsqr(matrix_layout, uplo, n, ncvt, nru, ncc, get_ptr(d), get_ptr(e), get_ptr(vt), ldvt, get_ptr(u), ldu, get_ptr(c), ldc);
            }
        break; }
        case dbdsqr: {
            int matrix_layout; char uplo; int n; int ncvt; int nru; int ncc; c_binary d; c_binary e; c_binary vt; int ldvt; c_binary u; int ldu; c_binary c; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc))
            ){
                LAPACKE_dbdsqr(matrix_layout, uplo, n, ncvt, nru, ncc, get_ptr(d), get_ptr(e), get_ptr(vt), ldvt, get_ptr(u), ldu, get_ptr(c), ldc);
            }
        break; }
        case cbdsqr: {
            int matrix_layout; char uplo; int n; int ncvt; int nru; int ncc; c_binary d; c_binary e; c_binary vt; int ldvt; c_binary u; int ldu; c_binary c; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc))
            ){
                LAPACKE_cbdsqr(matrix_layout, uplo, n, ncvt, nru, ncc, get_ptr(d), get_ptr(e), get_ptr(vt), ldvt, get_ptr(u), ldu, get_ptr(c), ldc);
            }
        break; }
        case zbdsqr: {
            int matrix_layout; char uplo; int n; int ncvt; int nru; int ncc; c_binary d; c_binary e; c_binary vt; int ldvt; c_binary u; int ldu; c_binary c; int ldc;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ncvt, &nru, &ncc, &d, &e, &vt, &ldvt, &u, &ldu, &c, &ldc))
            ){
                LAPACKE_zbdsqr(matrix_layout, uplo, n, ncvt, nru, ncc, get_ptr(d), get_ptr(e), get_ptr(vt), ldvt, get_ptr(u), ldu, get_ptr(c), ldc);
            }
        break; }
        case sdisna: {
            char job; int m; int n; cste_c_binary d; c_binary sep;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_int, e_cste_ptr, e_ptr, e_end}, &job, &m, &n, &d, &sep))
            ){
                LAPACKE_sdisna(job, m, n, get_cste_ptr(d), get_ptr(sep));
            }
        break; }
        case ddisna: {
            char job; int m; int n; cste_c_binary d; c_binary sep;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_int, e_cste_ptr, e_ptr, e_end}, &job, &m, &n, &d, &sep))
            ){
                LAPACKE_ddisna(job, m, n, get_cste_ptr(d), get_ptr(sep));
            }
        break; }
        case sgbbrd: {
            int matrix_layout; char vect; int m; int n; int ncc; int kl; int ku; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq; c_binary pt; int ldpt; c_binary c; int ldc;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &m, &n, &ncc, &kl, &ku, &ab, &ldab, &d, &e, &q, &ldq, &pt, &ldpt, &c, &ldc))
            ){
                LAPACKE_sgbbrd(matrix_layout, vect, m, n, ncc, kl, ku, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq, get_ptr(pt), ldpt, get_ptr(c), ldc);
            }
        break; }
        case dgbbrd: {
            int matrix_layout; char vect; int m; int n; int ncc; int kl; int ku; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq; c_binary pt; int ldpt; c_binary c; int ldc;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &m, &n, &ncc, &kl, &ku, &ab, &ldab, &d, &e, &q, &ldq, &pt, &ldpt, &c, &ldc))
            ){
                LAPACKE_dgbbrd(matrix_layout, vect, m, n, ncc, kl, ku, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq, get_ptr(pt), ldpt, get_ptr(c), ldc);
            }
        break; }
        case cgbbrd: {
            int matrix_layout; char vect; int m; int n; int ncc; int kl; int ku; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq; c_binary pt; int ldpt; c_binary c; int ldc;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &m, &n, &ncc, &kl, &ku, &ab, &ldab, &d, &e, &q, &ldq, &pt, &ldpt, &c, &ldc))
            ){
                LAPACKE_cgbbrd(matrix_layout, vect, m, n, ncc, kl, ku, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq, get_ptr(pt), ldpt, get_ptr(c), ldc);
            }
        break; }
        case zgbbrd: {
            int matrix_layout; char vect; int m; int n; int ncc; int kl; int ku; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq; c_binary pt; int ldpt; c_binary c; int ldc;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &m, &n, &ncc, &kl, &ku, &ab, &ldab, &d, &e, &q, &ldq, &pt, &ldpt, &c, &ldc))
            ){
                LAPACKE_zgbbrd(matrix_layout, vect, m, n, ncc, kl, ku, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq, get_ptr(pt), ldpt, get_ptr(c), ldc);
            }
        break; }
        case sgbcon: {
            int matrix_layout; char norm; int n; int kl; int ku; cste_c_binary ab; int ldab; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &kl, &ku, &ab, &ldab, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_sgbcon(matrix_layout, norm, n, kl, ku, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case dgbcon: {
            int matrix_layout; char norm; int n; int kl; int ku; cste_c_binary ab; int ldab; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &kl, &ku, &ab, &ldab, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_dgbcon(matrix_layout, norm, n, kl, ku, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case cgbcon: {
            int matrix_layout; char norm; int n; int kl; int ku; cste_c_binary ab; int ldab; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &kl, &ku, &ab, &ldab, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_cgbcon(matrix_layout, norm, n, kl, ku, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zgbcon: {
            int matrix_layout; char norm; int n; int kl; int ku; cste_c_binary ab; int ldab; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &kl, &ku, &ab, &ldab, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zgbcon(matrix_layout, norm, n, kl, ku, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case sgbequ: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_sgbequ(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case dgbequ: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_dgbequ(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case cgbequ: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_cgbequ(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case zgbequ: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_zgbequ(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case sgbequb: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_sgbequb(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case dgbequb: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_dgbequb(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case cgbequb: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_cgbequb(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case zgbequb: {
            int matrix_layout; int m; int n; int kl; int ku; cste_c_binary ab; int ldab; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_zgbequb(matrix_layout, m, n, kl, ku, get_cste_ptr(ab), ldab, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case sgbrfs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sgbrfs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dgbrfs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dgbrfs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cgbrfs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cgbrfs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zgbrfs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &afb, &ldafb, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zgbrfs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case sgbsv: {
            int matrix_layout; int n; int kl; int ku; int nrhs; c_binary ab; int ldab; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgbsv(matrix_layout, n, kl, ku, nrhs, get_ptr(ab), ldab, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case dgbsv: {
            int matrix_layout; int n; int kl; int ku; int nrhs; c_binary ab; int ldab; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgbsv(matrix_layout, n, kl, ku, nrhs, get_ptr(ab), ldab, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case cgbsv: {
            int matrix_layout; int n; int kl; int ku; int nrhs; c_binary ab; int ldab; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgbsv(matrix_layout, n, kl, ku, nrhs, get_ptr(ab), ldab, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zgbsv: {
            int matrix_layout; int n; int kl; int ku; int nrhs; c_binary ab; int ldab; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgbsv(matrix_layout, n, kl, ku, nrhs, get_ptr(ab), ldab, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case sgbtrf: {
            int matrix_layout; int m; int n; int kl; int ku; c_binary ab; int ldab; c_binary ipiv;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &ipiv))
            ){
                LAPACKE_sgbtrf(matrix_layout, m, n, kl, ku, get_ptr(ab), ldab, get_ptr(ipiv));
            }
        break; }
        case dgbtrf: {
            int matrix_layout; int m; int n; int kl; int ku; c_binary ab; int ldab; c_binary ipiv;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &ipiv))
            ){
                LAPACKE_dgbtrf(matrix_layout, m, n, kl, ku, get_ptr(ab), ldab, get_ptr(ipiv));
            }
        break; }
        case cgbtrf: {
            int matrix_layout; int m; int n; int kl; int ku; c_binary ab; int ldab; c_binary ipiv;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &ipiv))
            ){
                LAPACKE_cgbtrf(matrix_layout, m, n, kl, ku, get_ptr(ab), ldab, get_ptr(ipiv));
            }
        break; }
        case zgbtrf: {
            int matrix_layout; int m; int n; int kl; int ku; c_binary ab; int ldab; c_binary ipiv;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &kl, &ku, &ab, &ldab, &ipiv))
            ){
                LAPACKE_zgbtrf(matrix_layout, m, n, kl, ku, get_ptr(ab), ldab, get_ptr(ipiv));
            }
        break; }
        case sgbtrs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgbtrs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case dgbtrs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgbtrs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case cgbtrs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgbtrs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zgbtrs: {
            int matrix_layout; char trans; int n; int kl; int ku; int nrhs; cste_c_binary ab; int ldab; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &kl, &ku, &nrhs, &ab, &ldab, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgbtrs(matrix_layout, trans, n, kl, ku, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case sgebak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary scale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &scale, &m, &v, &ldv))
            ){
                LAPACKE_sgebak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(scale), m, get_ptr(v), ldv);
            }
        break; }
        case dgebak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary scale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &scale, &m, &v, &ldv))
            ){
                LAPACKE_dgebak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(scale), m, get_ptr(v), ldv);
            }
        break; }
        case cgebak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary scale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &scale, &m, &v, &ldv))
            ){
                LAPACKE_cgebak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(scale), m, get_ptr(v), ldv);
            }
        break; }
        case zgebak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary scale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &scale, &m, &v, &ldv))
            ){
                LAPACKE_zgebak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(scale), m, get_ptr(v), ldv);
            }
        break; }
        case sgebal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary ilo; c_binary ihi; c_binary scale;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &ilo, &ihi, &scale))
            ){
                LAPACKE_sgebal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(ilo), get_ptr(ihi), get_ptr(scale));
            }
        break; }
        case dgebal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary ilo; c_binary ihi; c_binary scale;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &ilo, &ihi, &scale))
            ){
                LAPACKE_dgebal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(ilo), get_ptr(ihi), get_ptr(scale));
            }
        break; }
        case cgebal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary ilo; c_binary ihi; c_binary scale;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &ilo, &ihi, &scale))
            ){
                LAPACKE_cgebal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(ilo), get_ptr(ihi), get_ptr(scale));
            }
        break; }
        case zgebal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary ilo; c_binary ihi; c_binary scale;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &ilo, &ihi, &scale))
            ){
                LAPACKE_zgebal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(ilo), get_ptr(ihi), get_ptr(scale));
            }
        break; }
        case sgebrd: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tauq; c_binary taup;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &d, &e, &tauq, &taup))
            ){
                LAPACKE_sgebrd(matrix_layout, m, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tauq), get_ptr(taup));
            }
        break; }
        case dgebrd: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tauq; c_binary taup;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &d, &e, &tauq, &taup))
            ){
                LAPACKE_dgebrd(matrix_layout, m, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tauq), get_ptr(taup));
            }
        break; }
        case cgebrd: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tauq; c_binary taup;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &d, &e, &tauq, &taup))
            ){
                LAPACKE_cgebrd(matrix_layout, m, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tauq), get_ptr(taup));
            }
        break; }
        case zgebrd: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tauq; c_binary taup;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &d, &e, &tauq, &taup))
            ){
                LAPACKE_zgebrd(matrix_layout, m, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tauq), get_ptr(taup));
            }
        break; }
        case sgecon: {
            int matrix_layout; char norm; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_sgecon(matrix_layout, norm, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case dgecon: {
            int matrix_layout; char norm; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_dgecon(matrix_layout, norm, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case cgecon: {
            int matrix_layout; char norm; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_cgecon(matrix_layout, norm, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zgecon: {
            int matrix_layout; char norm; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_zgecon(matrix_layout, norm, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case sgeequ: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_sgeequ(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case dgeequ: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_dgeequ(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case cgeequ: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_cgeequ(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case zgeequ: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_zgeequ(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case sgeequb: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_sgeequb(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case dgeequb: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_dgeequb(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case cgeequb: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_cgeequb(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case zgeequb: {
            int matrix_layout; int m; int n; cste_c_binary a; int lda; c_binary r; c_binary c; c_binary rowcnd; c_binary colcnd; c_binary amax;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &r, &c, &rowcnd, &colcnd, &amax))
            ){
                LAPACKE_zgeequb(matrix_layout, m, n, get_cste_ptr(a), lda, get_ptr(r), get_ptr(c), get_ptr(rowcnd), get_ptr(colcnd), get_ptr(amax));
            }
        break; }
        case sgeev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary wr; c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_sgeev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(wr), get_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        break; }
        case dgeev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary wr; c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_dgeev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(wr), get_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        break; }
        case cgeev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &w, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_cgeev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        break; }
        case zgeev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &w, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_zgeev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        break; }
        case sgeevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary wr; c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary scale; c_binary abnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv))
            ){
                LAPACKE_sgeevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(wr), get_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(scale), get_ptr(abnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        break; }
        case dgeevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary wr; c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary scale; c_binary abnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv))
            ){
                LAPACKE_dgeevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(wr), get_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(scale), get_ptr(abnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        break; }
        case cgeevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary scale; c_binary abnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &w, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv))
            ){
                LAPACKE_cgeevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(scale), get_ptr(abnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        break; }
        case zgeevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary scale; c_binary abnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &w, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &scale, &abnrm, &rconde, &rcondv))
            ){
                LAPACKE_zgeevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(scale), get_ptr(abnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        break; }
        case sgehrd: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_sgehrd(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case dgehrd: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_dgehrd(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case cgehrd: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_cgehrd(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case zgehrd: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_zgehrd(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case sgejsv: {
            int matrix_layout; char joba; char jobu; char jobv; char jobr; char jobt; char jobp; int m; int n; c_binary a; int lda; c_binary sva; c_binary u; int ldu; c_binary v; int ldv; c_binary stat; c_binary istat;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, &a, &lda, &sva, &u, &ldu, &v, &ldv, &stat, &istat))
            ){
                LAPACKE_sgejsv(matrix_layout, joba, jobu, jobv, jobr, jobt, jobp, m, n, get_ptr(a), lda, get_ptr(sva), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(stat), get_ptr(istat));
            }
        break; }
        case dgejsv: {
            int matrix_layout; char joba; char jobu; char jobv; char jobr; char jobt; char jobp; int m; int n; c_binary a; int lda; c_binary sva; c_binary u; int ldu; c_binary v; int ldv; c_binary stat; c_binary istat;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &joba, &jobu, &jobv, &jobr, &jobt, &jobp, &m, &n, &a, &lda, &sva, &u, &ldu, &v, &ldv, &stat, &istat))
            ){
                LAPACKE_dgejsv(matrix_layout, joba, jobu, jobv, jobr, jobt, jobp, m, n, get_ptr(a), lda, get_ptr(sva), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(stat), get_ptr(istat));
            }
        break; }
        case sgelqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgelqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case dgelqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgelqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case cgelqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgelqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case zgelqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgelqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case sgels: {
            int matrix_layout; char trans; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &trans, &m, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_sgels(matrix_layout, trans, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case dgels: {
            int matrix_layout; char trans; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &trans, &m, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dgels(matrix_layout, trans, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case cgels: {
            int matrix_layout; char trans; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &trans, &m, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_cgels(matrix_layout, trans, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case zgels: {
            int matrix_layout; char trans; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &trans, &m, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_zgels(matrix_layout, trans, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case sgelsd: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_sgelsd(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case dgelsd: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_dgelsd(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case cgelsd: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_cgelsd(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case zgelsd: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_zgelsd(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case sgelss: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_sgelss(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case dgelss: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_dgelss(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case cgelss: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_cgelss(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case zgelss: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary s; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &s, &rcond, &rank))
            ){
                LAPACKE_zgelss(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(s), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case sgelsy: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary jpvt; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &jpvt, &rcond, &rank))
            ){
                LAPACKE_sgelsy(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(jpvt), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case dgelsy: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary jpvt; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &jpvt, &rcond, &rank))
            ){
                LAPACKE_dgelsy(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(jpvt), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case cgelsy: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary jpvt; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &jpvt, &rcond, &rank))
            ){
                LAPACKE_cgelsy(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(jpvt), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case zgelsy: {
            int matrix_layout; int m; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb; c_binary jpvt; cste_c_binary rcond; c_binary rank;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &nrhs, &a, &lda, &b, &ldb, &jpvt, &rcond, &rank))
            ){
                LAPACKE_zgelsy(matrix_layout, m, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(jpvt), get_cste_double(rcond), get_ptr(rank));
            }
        break; }
        case sgeqlf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgeqlf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case dgeqlf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgeqlf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case cgeqlf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgeqlf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case zgeqlf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgeqlf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case sgeqp3: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_sgeqp3(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        break; }
        case dgeqp3: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_dgeqp3(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        break; }
        case cgeqp3: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_cgeqp3(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        break; }
        case zgeqp3: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_zgeqp3(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        break; }
        case sgeqpf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_sgeqpf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        break; }
        case dgeqpf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_dgeqpf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        break; }
        case cgeqpf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_cgeqpf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        break; }
        case zgeqpf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary jpvt; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &jpvt, &tau))
            ){
                LAPACKE_zgeqpf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(jpvt), get_ptr(tau));
            }
        break; }
        case sgeqrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgeqrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case dgeqrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgeqrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case cgeqrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgeqrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case zgeqrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgeqrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case sgeqrfp: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgeqrfp(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case dgeqrfp: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgeqrfp(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case cgeqrfp: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgeqrfp(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case zgeqrfp: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgeqrfp(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case sgerfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sgerfs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dgerfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dgerfs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cgerfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cgerfs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zgerfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zgerfs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case sgerqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_sgerqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case dgerqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dgerqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case cgerqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_cgerqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case zgerqf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_zgerqf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case sgesdd: {
            int matrix_layout; char jobz; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt))
            ){
                LAPACKE_sgesdd(matrix_layout, jobz, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt);
            }
        break; }
        case dgesdd: {
            int matrix_layout; char jobz; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt))
            ){
                LAPACKE_dgesdd(matrix_layout, jobz, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt);
            }
        break; }
        case cgesdd: {
            int matrix_layout; char jobz; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt))
            ){
                LAPACKE_cgesdd(matrix_layout, jobz, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt);
            }
        break; }
        case zgesdd: {
            int matrix_layout; char jobz; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt))
            ){
                LAPACKE_zgesdd(matrix_layout, jobz, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt);
            }
        break; }
        case sgesv: {
            int matrix_layout; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgesv(matrix_layout, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case dgesv: {
            int matrix_layout; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgesv(matrix_layout, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case cgesv: {
            int matrix_layout; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgesv(matrix_layout, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zgesv: {
            int matrix_layout; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgesv(matrix_layout, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case sgesvd: {
            int matrix_layout; char jobu; char jobvt; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt; c_binary superb;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &superb))
            ){
                LAPACKE_sgesvd(matrix_layout, jobu, jobvt, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(superb));
            }
        break; }
        case dgesvd: {
            int matrix_layout; char jobu; char jobvt; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt; c_binary superb;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &superb))
            ){
                LAPACKE_dgesvd(matrix_layout, jobu, jobvt, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(superb));
            }
        break; }
        case cgesvd: {
            int matrix_layout; char jobu; char jobvt; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt; c_binary superb;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &superb))
            ){
                LAPACKE_cgesvd(matrix_layout, jobu, jobvt, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(superb));
            }
        break; }
        case zgesvd: {
            int matrix_layout; char jobu; char jobvt; int m; int n; c_binary a; int lda; c_binary s; c_binary u; int ldu; c_binary vt; int ldvt; c_binary superb;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &superb))
            ){
                LAPACKE_zgesvd(matrix_layout, jobu, jobvt, m, n, get_ptr(a), lda, get_ptr(s), get_ptr(u), ldu, get_ptr(vt), ldvt, get_ptr(superb));
            }
        break; }
        case sgesvj: {
            int matrix_layout; char joba; char jobu; char jobv; int m; int n; c_binary a; int lda; c_binary sva; int mv; c_binary v; int ldv; c_binary stat;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &joba, &jobu, &jobv, &m, &n, &a, &lda, &sva, &mv, &v, &ldv, &stat))
            ){
                LAPACKE_sgesvj(matrix_layout, joba, jobu, jobv, m, n, get_ptr(a), lda, get_ptr(sva), mv, get_ptr(v), ldv, get_ptr(stat));
            }
        break; }
        case dgesvj: {
            int matrix_layout; char joba; char jobu; char jobv; int m; int n; c_binary a; int lda; c_binary sva; int mv; c_binary v; int ldv; c_binary stat;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &joba, &jobu, &jobv, &m, &n, &a, &lda, &sva, &mv, &v, &ldv, &stat))
            ){
                LAPACKE_dgesvj(matrix_layout, joba, jobu, jobv, m, n, get_ptr(a), lda, get_ptr(sva), mv, get_ptr(v), ldv, get_ptr(stat));
            }
        break; }
        case sgetrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_sgetrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case dgetrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_dgetrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case cgetrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_cgetrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case zgetrf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zgetrf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case sgetri: {
            int matrix_layout; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_sgetri(matrix_layout, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case dgetri: {
            int matrix_layout; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_dgetri(matrix_layout, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case cgetri: {
            int matrix_layout; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_cgetri(matrix_layout, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case zgetri: {
            int matrix_layout; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zgetri(matrix_layout, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case sgetrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgetrs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case dgetrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgetrs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case cgetrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgetrs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zgetrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgetrs(matrix_layout, trans, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case sggbak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary lscale; cste_c_binary rscale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &lscale, &rscale, &m, &v, &ldv))
            ){
                LAPACKE_sggbak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(lscale), get_cste_ptr(rscale), m, get_ptr(v), ldv);
            }
        break; }
        case dggbak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary lscale; cste_c_binary rscale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &lscale, &rscale, &m, &v, &ldv))
            ){
                LAPACKE_dggbak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(lscale), get_cste_ptr(rscale), m, get_ptr(v), ldv);
            }
        break; }
        case cggbak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary lscale; cste_c_binary rscale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &lscale, &rscale, &m, &v, &ldv))
            ){
                LAPACKE_cggbak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(lscale), get_cste_ptr(rscale), m, get_ptr(v), ldv);
            }
        break; }
        case zggbak: {
            int matrix_layout; char job; char side; int n; int ilo; int ihi; cste_c_binary lscale; cste_c_binary rscale; int m; c_binary v; int ldv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &side, &n, &ilo, &ihi, &lscale, &rscale, &m, &v, &ldv))
            ){
                LAPACKE_zggbak(matrix_layout, job, side, n, ilo, ihi, get_cste_ptr(lscale), get_cste_ptr(rscale), m, get_ptr(v), ldv);
            }
        break; }
        case sggbal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary b; int ldb; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &b, &ldb, &ilo, &ihi, &lscale, &rscale))
            ){
                LAPACKE_sggbal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale));
            }
        break; }
        case dggbal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary b; int ldb; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &b, &ldb, &ilo, &ihi, &lscale, &rscale))
            ){
                LAPACKE_dggbal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale));
            }
        break; }
        case cggbal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary b; int ldb; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &b, &ldb, &ilo, &ihi, &lscale, &rscale))
            ){
                LAPACKE_cggbal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale));
            }
        break; }
        case zggbal: {
            int matrix_layout; char job; int n; c_binary a; int lda; c_binary b; int ldb; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &n, &a, &lda, &b, &ldb, &ilo, &ihi, &lscale, &rscale))
            ){
                LAPACKE_zggbal(matrix_layout, job, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale));
            }
        break; }
        case sggev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_sggev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        break; }
        case dggev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_dggev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        break; }
        case cggev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &b, &ldb, &alpha, &beta, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_cggev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        break; }
        case zggev: {
            int matrix_layout; char jobvl; char jobvr; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobvl, &jobvr, &n, &a, &lda, &b, &ldb, &alpha, &beta, &vl, &ldvl, &vr, &ldvr))
            ){
                LAPACKE_zggev(matrix_layout, jobvl, jobvr, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr);
            }
        break; }
        case sggevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale; c_binary abnrm; c_binary bbnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 25? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &lscale, &rscale, &abnrm, &bbnrm, &rconde, &rcondv))
            ){
                LAPACKE_sggevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale), get_ptr(abnrm), get_ptr(bbnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        break; }
        case dggevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale; c_binary abnrm; c_binary bbnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 25? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &lscale, &rscale, &abnrm, &bbnrm, &rconde, &rcondv))
            ){
                LAPACKE_dggevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale), get_ptr(abnrm), get_ptr(bbnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        break; }
        case cggevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale; c_binary abnrm; c_binary bbnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &b, &ldb, &alpha, &beta, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &lscale, &rscale, &abnrm, &bbnrm, &rconde, &rcondv))
            ){
                LAPACKE_cggevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale), get_ptr(abnrm), get_ptr(bbnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        break; }
        case zggevx: {
            int matrix_layout; char balanc; char jobvl; char jobvr; char sense; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary vl; int ldvl; c_binary vr; int ldvr; c_binary ilo; c_binary ihi; c_binary lscale; c_binary rscale; c_binary abnrm; c_binary bbnrm; c_binary rconde; c_binary rcondv;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &balanc, &jobvl, &jobvr, &sense, &n, &a, &lda, &b, &ldb, &alpha, &beta, &vl, &ldvl, &vr, &ldvr, &ilo, &ihi, &lscale, &rscale, &abnrm, &bbnrm, &rconde, &rcondv))
            ){
                LAPACKE_zggevx(matrix_layout, balanc, jobvl, jobvr, sense, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(vl), ldvl, get_ptr(vr), ldvr, get_ptr(ilo), get_ptr(ihi), get_ptr(lscale), get_ptr(rscale), get_ptr(abnrm), get_ptr(bbnrm), get_ptr(rconde), get_ptr(rcondv));
            }
        break; }
        case sggglm: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary b; int ldb; c_binary d; c_binary x; c_binary y;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &b, &ldb, &d, &x, &y))
            ){
                LAPACKE_sggglm(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(d), get_ptr(x), get_ptr(y));
            }
        break; }
        case dggglm: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary b; int ldb; c_binary d; c_binary x; c_binary y;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &b, &ldb, &d, &x, &y))
            ){
                LAPACKE_dggglm(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(d), get_ptr(x), get_ptr(y));
            }
        break; }
        case cggglm: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary b; int ldb; c_binary d; c_binary x; c_binary y;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &b, &ldb, &d, &x, &y))
            ){
                LAPACKE_cggglm(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(d), get_ptr(x), get_ptr(y));
            }
        break; }
        case zggglm: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary b; int ldb; c_binary d; c_binary x; c_binary y;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &b, &ldb, &d, &x, &y))
            ){
                LAPACKE_zggglm(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(d), get_ptr(x), get_ptr(y));
            }
        break; }
        case sgghrd: {
            int matrix_layout; char compq; char compz; int n; int ilo; int ihi; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &compq, &compz, &n, &ilo, &ihi, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_sgghrd(matrix_layout, compq, compz, n, ilo, ihi, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz);
            }
        break; }
        case dgghrd: {
            int matrix_layout; char compq; char compz; int n; int ilo; int ihi; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &compq, &compz, &n, &ilo, &ihi, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_dgghrd(matrix_layout, compq, compz, n, ilo, ihi, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz);
            }
        break; }
        case cgghrd: {
            int matrix_layout; char compq; char compz; int n; int ilo; int ihi; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &compq, &compz, &n, &ilo, &ihi, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_cgghrd(matrix_layout, compq, compz, n, ilo, ihi, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz);
            }
        break; }
        case zgghrd: {
            int matrix_layout; char compq; char compz; int n; int ilo; int ihi; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &compq, &compz, &n, &ilo, &ihi, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_zgghrd(matrix_layout, compq, compz, n, ilo, ihi, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz);
            }
        break; }
        case sgglse: {
            int matrix_layout; int m; int n; int p; c_binary a; int lda; c_binary b; int ldb; c_binary c; c_binary d; c_binary x;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &p, &a, &lda, &b, &ldb, &c, &d, &x))
            ){
                LAPACKE_sgglse(matrix_layout, m, n, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(c), get_ptr(d), get_ptr(x));
            }
        break; }
        case dgglse: {
            int matrix_layout; int m; int n; int p; c_binary a; int lda; c_binary b; int ldb; c_binary c; c_binary d; c_binary x;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &p, &a, &lda, &b, &ldb, &c, &d, &x))
            ){
                LAPACKE_dgglse(matrix_layout, m, n, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(c), get_ptr(d), get_ptr(x));
            }
        break; }
        case cgglse: {
            int matrix_layout; int m; int n; int p; c_binary a; int lda; c_binary b; int ldb; c_binary c; c_binary d; c_binary x;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &p, &a, &lda, &b, &ldb, &c, &d, &x))
            ){
                LAPACKE_cgglse(matrix_layout, m, n, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(c), get_ptr(d), get_ptr(x));
            }
        break; }
        case zgglse: {
            int matrix_layout; int m; int n; int p; c_binary a; int lda; c_binary b; int ldb; c_binary c; c_binary d; c_binary x;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &m, &n, &p, &a, &lda, &b, &ldb, &c, &d, &x))
            ){
                LAPACKE_zgglse(matrix_layout, m, n, p, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(c), get_ptr(d), get_ptr(x));
            }
        break; }
        case sggqrf: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_sggqrf(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        break; }
        case dggqrf: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_dggqrf(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        break; }
        case cggqrf: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_cggqrf(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        break; }
        case zggqrf: {
            int matrix_layout; int n; int m; int p; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &m, &p, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_zggqrf(matrix_layout, n, m, p, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        break; }
        case sggrqf: {
            int matrix_layout; int m; int p; int n; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &p, &n, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_sggrqf(matrix_layout, m, p, n, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        break; }
        case dggrqf: {
            int matrix_layout; int m; int p; int n; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &p, &n, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_dggrqf(matrix_layout, m, p, n, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        break; }
        case cggrqf: {
            int matrix_layout; int m; int p; int n; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &p, &n, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_cggrqf(matrix_layout, m, p, n, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        break; }
        case zggrqf: {
            int matrix_layout; int m; int p; int n; c_binary a; int lda; c_binary taua; c_binary b; int ldb; c_binary taub;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &p, &n, &a, &lda, &taua, &b, &ldb, &taub))
            ){
                LAPACKE_zggrqf(matrix_layout, m, p, n, get_ptr(a), lda, get_ptr(taua), get_ptr(b), ldb, get_ptr(taub));
            }
        break; }
        case sggsvd: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int n; int p; c_binary k; c_binary l; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary iwork;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &n, &p, &k, &l, &a, &lda, &b, &ldb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &iwork))
            ){
                LAPACKE_sggsvd(matrix_layout, jobu, jobv, jobq, m, n, p, get_ptr(k), get_ptr(l), get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(iwork));
            }
        break; }
        case dggsvd: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int n; int p; c_binary k; c_binary l; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary iwork;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &n, &p, &k, &l, &a, &lda, &b, &ldb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &iwork))
            ){
                LAPACKE_dggsvd(matrix_layout, jobu, jobv, jobq, m, n, p, get_ptr(k), get_ptr(l), get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(iwork));
            }
        break; }
        case cggsvd: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int n; int p; c_binary k; c_binary l; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary iwork;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &n, &p, &k, &l, &a, &lda, &b, &ldb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &iwork))
            ){
                LAPACKE_cggsvd(matrix_layout, jobu, jobv, jobq, m, n, p, get_ptr(k), get_ptr(l), get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(iwork));
            }
        break; }
        case zggsvd: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int n; int p; c_binary k; c_binary l; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary iwork;
            
            if( !(error = narg == 22? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &n, &p, &k, &l, &a, &lda, &b, &ldb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &iwork))
            ){
                LAPACKE_zggsvd(matrix_layout, jobu, jobv, jobq, m, n, p, get_ptr(k), get_ptr(l), get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(iwork));
            }
        break; }
        case sggsvp: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary k; c_binary l; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &a, &lda, &b, &ldb, &tola, &tolb, &k, &l, &u, &ldu, &v, &ldv, &q, &ldq))
            ){
                LAPACKE_sggsvp(matrix_layout, jobu, jobv, jobq, m, p, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(k), get_ptr(l), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq);
            }
        break; }
        case dggsvp: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary k; c_binary l; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &a, &lda, &b, &ldb, &tola, &tolb, &k, &l, &u, &ldu, &v, &ldv, &q, &ldq))
            ){
                LAPACKE_dggsvp(matrix_layout, jobu, jobv, jobq, m, p, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(k), get_ptr(l), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq);
            }
        break; }
        case cggsvp: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary k; c_binary l; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &a, &lda, &b, &ldb, &tola, &tolb, &k, &l, &u, &ldu, &v, &ldv, &q, &ldq))
            ){
                LAPACKE_cggsvp(matrix_layout, jobu, jobv, jobq, m, p, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(k), get_ptr(l), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq);
            }
        break; }
        case zggsvp: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary k; c_binary l; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &a, &lda, &b, &ldb, &tola, &tolb, &k, &l, &u, &ldu, &v, &ldv, &q, &ldq))
            ){
                LAPACKE_zggsvp(matrix_layout, jobu, jobv, jobq, m, p, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(k), get_ptr(l), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq);
            }
        break; }
        case sgtcon: {
            char norm; int n; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &norm, &n, &dl, &d, &du, &du2, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_sgtcon(norm, n, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case dgtcon: {
            char norm; int n; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &norm, &n, &dl, &d, &du, &du2, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_dgtcon(norm, n, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case cgtcon: {
            char norm; int n; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &norm, &n, &dl, &d, &du, &du2, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_cgtcon(norm, n, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zgtcon: {
            char norm; int n; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &norm, &n, &dl, &d, &du, &du2, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zgtcon(norm, n, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case sgtrfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary dlf; cste_c_binary df; cste_c_binary duf; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sgtrfs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(dlf), get_cste_ptr(df), get_cste_ptr(duf), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dgtrfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary dlf; cste_c_binary df; cste_c_binary duf; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dgtrfs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(dlf), get_cste_ptr(df), get_cste_ptr(duf), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cgtrfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary dlf; cste_c_binary df; cste_c_binary duf; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cgtrfs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(dlf), get_cste_ptr(df), get_cste_ptr(duf), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zgtrfs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary dlf; cste_c_binary df; cste_c_binary duf; cste_c_binary du2; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zgtrfs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(dlf), get_cste_ptr(df), get_cste_ptr(duf), get_cste_ptr(du2), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case sgtsv: {
            int matrix_layout; int n; int nrhs; c_binary dl; c_binary d; c_binary du; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &dl, &d, &du, &b, &ldb))
            ){
                LAPACKE_sgtsv(matrix_layout, n, nrhs, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(b), ldb);
            }
        break; }
        case dgtsv: {
            int matrix_layout; int n; int nrhs; c_binary dl; c_binary d; c_binary du; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &dl, &d, &du, &b, &ldb))
            ){
                LAPACKE_dgtsv(matrix_layout, n, nrhs, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(b), ldb);
            }
        break; }
        case cgtsv: {
            int matrix_layout; int n; int nrhs; c_binary dl; c_binary d; c_binary du; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &dl, &d, &du, &b, &ldb))
            ){
                LAPACKE_cgtsv(matrix_layout, n, nrhs, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(b), ldb);
            }
        break; }
        case zgtsv: {
            int matrix_layout; int n; int nrhs; c_binary dl; c_binary d; c_binary du; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &dl, &d, &du, &b, &ldb))
            ){
                LAPACKE_zgtsv(matrix_layout, n, nrhs, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(b), ldb);
            }
        break; }
        case sgtsvx: {
            int matrix_layout; char fact; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; c_binary dlf; c_binary df; c_binary duf; c_binary du2; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_sgtsvx(matrix_layout, fact, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_ptr(dlf), get_ptr(df), get_ptr(duf), get_ptr(du2), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dgtsvx: {
            int matrix_layout; char fact; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; c_binary dlf; c_binary df; c_binary duf; c_binary du2; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_dgtsvx(matrix_layout, fact, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_ptr(dlf), get_ptr(df), get_ptr(duf), get_ptr(du2), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cgtsvx: {
            int matrix_layout; char fact; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; c_binary dlf; c_binary df; c_binary duf; c_binary du2; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_cgtsvx(matrix_layout, fact, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_ptr(dlf), get_ptr(df), get_ptr(duf), get_ptr(du2), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zgtsvx: {
            int matrix_layout; char fact; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; c_binary dlf; c_binary df; c_binary duf; c_binary du2; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &trans, &n, &nrhs, &dl, &d, &du, &dlf, &df, &duf, &du2, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zgtsvx(matrix_layout, fact, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_ptr(dlf), get_ptr(df), get_ptr(duf), get_ptr(du2), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case sgttrf: {
            int n; c_binary dl; c_binary d; c_binary du; c_binary du2; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &n, &dl, &d, &du, &du2, &ipiv))
            ){
                LAPACKE_sgttrf(n, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(du2), get_ptr(ipiv));
            }
        break; }
        case dgttrf: {
            int n; c_binary dl; c_binary d; c_binary du; c_binary du2; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &n, &dl, &d, &du, &du2, &ipiv))
            ){
                LAPACKE_dgttrf(n, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(du2), get_ptr(ipiv));
            }
        break; }
        case cgttrf: {
            int n; c_binary dl; c_binary d; c_binary du; c_binary du2; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &n, &dl, &d, &du, &du2, &ipiv))
            ){
                LAPACKE_cgttrf(n, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(du2), get_ptr(ipiv));
            }
        break; }
        case zgttrf: {
            int n; c_binary dl; c_binary d; c_binary du; c_binary du2; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &n, &dl, &d, &du, &du2, &ipiv))
            ){
                LAPACKE_zgttrf(n, get_ptr(dl), get_ptr(d), get_ptr(du), get_ptr(du2), get_ptr(ipiv));
            }
        break; }
        case sgttrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &du2, &ipiv, &b, &ldb))
            ){
                LAPACKE_sgttrs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case dgttrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &du2, &ipiv, &b, &ldb))
            ){
                LAPACKE_dgttrs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case cgttrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &du2, &ipiv, &b, &ldb))
            ){
                LAPACKE_cgttrs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zgttrs: {
            int matrix_layout; char trans; int n; int nrhs; cste_c_binary dl; cste_c_binary d; cste_c_binary du; cste_c_binary du2; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &trans, &n, &nrhs, &dl, &d, &du, &du2, &ipiv, &b, &ldb))
            ){
                LAPACKE_zgttrs(matrix_layout, trans, n, nrhs, get_cste_ptr(dl), get_cste_ptr(d), get_cste_ptr(du), get_cste_ptr(du2), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case chbev: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_chbev(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case zhbev: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_zhbev(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case chbevd: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_chbevd(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case zhbevd: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_zhbevd(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case chbevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int kd; c_binary ab; int ldab; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &kd, &ab, &ldab, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chbevx(matrix_layout, jobz, range, uplo, n, kd, get_ptr(ab), ldab, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case zhbevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int kd; c_binary ab; int ldab; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &kd, &ab, &ldab, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhbevx(matrix_layout, jobz, range, uplo, n, kd, get_ptr(ab), ldab, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case chbgst: {
            int matrix_layout; char vect; char uplo; int n; int ka; int kb; c_binary ab; int ldab; cste_c_binary bb; int ldbb; c_binary x; int ldx;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &x, &ldx))
            ){
                LAPACKE_chbgst(matrix_layout, vect, uplo, n, ka, kb, get_ptr(ab), ldab, get_cste_ptr(bb), ldbb, get_ptr(x), ldx);
            }
        break; }
        case zhbgst: {
            int matrix_layout; char vect; char uplo; int n; int ka; int kb; c_binary ab; int ldab; cste_c_binary bb; int ldbb; c_binary x; int ldx;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &x, &ldx))
            ){
                LAPACKE_zhbgst(matrix_layout, vect, uplo, n, ka, kb, get_ptr(ab), ldab, get_cste_ptr(bb), ldbb, get_ptr(x), ldx);
            }
        break; }
        case chbgv: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_chbgv(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case zhbgv: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_zhbgv(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case chbgvd: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_chbgvd(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case zhbgvd: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_zhbgvd(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case chbgvx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chbgvx(matrix_layout, jobz, range, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case zhbgvx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhbgvx(matrix_layout, jobz, range, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case chbtrd: {
            int matrix_layout; char vect; char uplo; int n; int kd; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &kd, &ab, &ldab, &d, &e, &q, &ldq))
            ){
                LAPACKE_chbtrd(matrix_layout, vect, uplo, n, kd, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq);
            }
        break; }
        case zhbtrd: {
            int matrix_layout; char vect; char uplo; int n; int kd; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &kd, &ab, &ldab, &d, &e, &q, &ldq))
            ){
                LAPACKE_zhbtrd(matrix_layout, vect, uplo, n, kd, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq);
            }
        break; }
        case checon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_checon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zhecon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zhecon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case cheequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_cheequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case zheequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_zheequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case cheev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_cheev(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        break; }
        case zheev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_zheev(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        break; }
        case cheevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_cheevd(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        break; }
        case zheevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_zheevd(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        break; }
        case cheevr: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_cheevr(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case zheevr: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_zheevr(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case cheevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_cheevx(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case zheevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zheevx(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case chegst: {
            int matrix_layout; int itype; char uplo; int n; c_binary a; int lda; cste_c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_end}, &matrix_layout, &itype, &uplo, &n, &a, &lda, &b, &ldb))
            ){
                LAPACKE_chegst(matrix_layout, itype, uplo, n, get_ptr(a), lda, get_cste_ptr(b), ldb);
            }
        break; }
        case zhegst: {
            int matrix_layout; int itype; char uplo; int n; c_binary a; int lda; cste_c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_end}, &matrix_layout, &itype, &uplo, &n, &a, &lda, &b, &ldb))
            ){
                LAPACKE_zhegst(matrix_layout, itype, uplo, n, get_ptr(a), lda, get_cste_ptr(b), ldb);
            }
        break; }
        case chegv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_chegv(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        break; }
        case zhegv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_zhegv(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        break; }
        case chegvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_chegvd(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        break; }
        case zhegvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_zhegvd(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        break; }
        case chegvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &a, &lda, &b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chegvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case zhegvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &a, &lda, &b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhegvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case cherfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cherfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zherfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zherfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case chesv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_chesv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zhesv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zhesv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case chesvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_chesvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zhesvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zhesvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case chetrd: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &d, &e, &tau))
            ){
                LAPACKE_chetrd(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        break; }
        case zhetrd: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &d, &e, &tau))
            ){
                LAPACKE_zhetrd(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        break; }
        case chetrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_chetrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case zhetrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zhetrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case chetri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_chetri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case zhetri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zhetri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case chetrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_chetrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zhetrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zhetrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case chfrk: {
            int matrix_layout; char transr; char uplo; char trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c))
            ){
                LAPACKE_chfrk(matrix_layout, transr, uplo, trans, n, k, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_double(beta), get_ptr(c));
            }
        break; }
        case zhfrk: {
            int matrix_layout; char transr; char uplo; char trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c))
            ){
                LAPACKE_zhfrk(matrix_layout, transr, uplo, trans, n, k, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_double(beta), get_ptr(c));
            }
        break; }
        case shgeqz: {
            int matrix_layout; char job; char compq; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary t; int ldt; c_binary alphar; c_binary alphai; c_binary beta; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &compq, &compz, &n, &ilo, &ihi, &h, &ldh, &t, &ldt, &alphar, &alphai, &beta, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_shgeqz(matrix_layout, job, compq, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(t), ldt, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz);
            }
        break; }
        case dhgeqz: {
            int matrix_layout; char job; char compq; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary t; int ldt; c_binary alphar; c_binary alphai; c_binary beta; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &compq, &compz, &n, &ilo, &ihi, &h, &ldh, &t, &ldt, &alphar, &alphai, &beta, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_dhgeqz(matrix_layout, job, compq, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(t), ldt, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz);
            }
        break; }
        case chgeqz: {
            int matrix_layout; char job; char compq; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary t; int ldt; c_binary alpha; c_binary beta; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &compq, &compz, &n, &ilo, &ihi, &h, &ldh, &t, &ldt, &alpha, &beta, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_chgeqz(matrix_layout, job, compq, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(t), ldt, get_ptr(alpha), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz);
            }
        break; }
        case zhgeqz: {
            int matrix_layout; char job; char compq; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary t; int ldt; c_binary alpha; c_binary beta; c_binary q; int ldq; c_binary z; int ldz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &job, &compq, &compz, &n, &ilo, &ihi, &h, &ldh, &t, &ldt, &alpha, &beta, &q, &ldq, &z, &ldz))
            ){
                LAPACKE_zhgeqz(matrix_layout, job, compq, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(t), ldt, get_ptr(alpha), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz);
            }
        break; }
        case chpcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_chpcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zhpcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zhpcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case chpev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_chpev(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case zhpev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_zhpev(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case chpevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_chpevd(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case zhpevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_zhpevd(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case chpevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary ap; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ap, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chpevx(matrix_layout, jobz, range, uplo, n, get_ptr(ap), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case zhpevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary ap; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ap, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhpevx(matrix_layout, jobz, range, uplo, n, get_ptr(ap), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case chpgst: {
            int matrix_layout; int itype; char uplo; int n; c_binary ap; cste_c_binary bp;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &itype, &uplo, &n, &ap, &bp))
            ){
                LAPACKE_chpgst(matrix_layout, itype, uplo, n, get_ptr(ap), get_cste_ptr(bp));
            }
        break; }
        case zhpgst: {
            int matrix_layout; int itype; char uplo; int n; c_binary ap; cste_c_binary bp;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &itype, &uplo, &n, &ap, &bp))
            ){
                LAPACKE_zhpgst(matrix_layout, itype, uplo, n, get_ptr(ap), get_cste_ptr(bp));
            }
        break; }
        case chpgv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_chpgv(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case zhpgv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_zhpgv(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case chpgvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_chpgvd(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case zhpgvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_zhpgvd(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case chpgvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary ap; c_binary bp; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &ap, &bp, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_chpgvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(ap), get_ptr(bp), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case zhpgvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary ap; c_binary bp; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &ap, &bp, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_zhpgvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(ap), get_ptr(bp), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case chprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_chprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zhprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zhprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case chpsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_chpsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zhpsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_zhpsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case chpsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_chpsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zhpsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zhpsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case chptrd: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &d, &e, &tau))
            ){
                LAPACKE_chptrd(matrix_layout, uplo, n, get_ptr(ap), get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        break; }
        case zhptrd: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &d, &e, &tau))
            ){
                LAPACKE_zhptrd(matrix_layout, uplo, n, get_ptr(ap), get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        break; }
        case chptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_chptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        break; }
        case zhptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_zhptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        break; }
        case chptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_chptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        break; }
        case zhptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_zhptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        break; }
        case chptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_chptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zhptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_zhptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case shsein: {
            int matrix_layout; char job; char eigsrc; char initv; c_binary select; int n; cste_c_binary h; int ldh; c_binary wr; cste_c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m; c_binary ifaill; c_binary ifailr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &eigsrc, &initv, &select, &n, &h, &ldh, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &mm, &m, &ifaill, &ifailr))
            ){
                LAPACKE_shsein(matrix_layout, job, eigsrc, initv, get_ptr(select), n, get_cste_ptr(h), ldh, get_ptr(wr), get_cste_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m), get_ptr(ifaill), get_ptr(ifailr));
            }
        break; }
        case dhsein: {
            int matrix_layout; char job; char eigsrc; char initv; c_binary select; int n; cste_c_binary h; int ldh; c_binary wr; cste_c_binary wi; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m; c_binary ifaill; c_binary ifailr;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &eigsrc, &initv, &select, &n, &h, &ldh, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &mm, &m, &ifaill, &ifailr))
            ){
                LAPACKE_dhsein(matrix_layout, job, eigsrc, initv, get_ptr(select), n, get_cste_ptr(h), ldh, get_ptr(wr), get_cste_ptr(wi), get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m), get_ptr(ifaill), get_ptr(ifailr));
            }
        break; }
        case chsein: {
            int matrix_layout; char job; char eigsrc; char initv; cste_c_binary select; int n; cste_c_binary h; int ldh; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m; c_binary ifaill; c_binary ifailr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &eigsrc, &initv, &select, &n, &h, &ldh, &w, &vl, &ldvl, &vr, &ldvr, &mm, &m, &ifaill, &ifailr))
            ){
                LAPACKE_chsein(matrix_layout, job, eigsrc, initv, get_cste_ptr(select), n, get_cste_ptr(h), ldh, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m), get_ptr(ifaill), get_ptr(ifailr));
            }
        break; }
        case zhsein: {
            int matrix_layout; char job; char eigsrc; char initv; cste_c_binary select; int n; cste_c_binary h; int ldh; c_binary w; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m; c_binary ifaill; c_binary ifailr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &eigsrc, &initv, &select, &n, &h, &ldh, &w, &vl, &ldvl, &vr, &ldvr, &mm, &m, &ifaill, &ifailr))
            ){
                LAPACKE_zhsein(matrix_layout, job, eigsrc, initv, get_cste_ptr(select), n, get_cste_ptr(h), ldh, get_ptr(w), get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m), get_ptr(ifaill), get_ptr(ifailr));
            }
        break; }
        case shseqr: {
            int matrix_layout; char job; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary wr; c_binary wi; c_binary z; int ldz;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &job, &compz, &n, &ilo, &ihi, &h, &ldh, &wr, &wi, &z, &ldz))
            ){
                LAPACKE_shseqr(matrix_layout, job, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(wr), get_ptr(wi), get_ptr(z), ldz);
            }
        break; }
        case dhseqr: {
            int matrix_layout; char job; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary wr; c_binary wi; c_binary z; int ldz;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &job, &compz, &n, &ilo, &ihi, &h, &ldh, &wr, &wi, &z, &ldz))
            ){
                LAPACKE_dhseqr(matrix_layout, job, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(wr), get_ptr(wi), get_ptr(z), ldz);
            }
        break; }
        case chseqr: {
            int matrix_layout; char job; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &job, &compz, &n, &ilo, &ihi, &h, &ldh, &w, &z, &ldz))
            ){
                LAPACKE_chseqr(matrix_layout, job, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case zhseqr: {
            int matrix_layout; char job; char compz; int n; int ilo; int ihi; c_binary h; int ldh; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &job, &compz, &n, &ilo, &ihi, &h, &ldh, &w, &z, &ldz))
            ){
                LAPACKE_zhseqr(matrix_layout, job, compz, n, ilo, ihi, get_ptr(h), ldh, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case sopgtr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary tau; c_binary q; int ldq;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &tau, &q, &ldq))
            ){
                LAPACKE_sopgtr(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(q), ldq);
            }
        break; }
        case dopgtr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary tau; c_binary q; int ldq;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &tau, &q, &ldq))
            ){
                LAPACKE_dopgtr(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(q), ldq);
            }
        break; }
        case sopmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary ap; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &ap, &tau, &c, &ldc))
            ){
                LAPACKE_sopmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case dopmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary ap; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &ap, &tau, &c, &ldc))
            ){
                LAPACKE_dopmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case sorgbr: {
            int matrix_layout; char vect; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &vect, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorgbr(matrix_layout, vect, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case dorgbr: {
            int matrix_layout; char vect; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &vect, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorgbr(matrix_layout, vect, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case sorghr: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_sorghr(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case dorghr: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_dorghr(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case sorglq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorglq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case dorglq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorglq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case sorgql: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorgql(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case dorgql: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorgql(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case sorgqr: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorgqr(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case dorgqr: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorgqr(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case sorgrq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_sorgrq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case dorgrq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_dorgrq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case sorgtr: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &tau))
            ){
                LAPACKE_sorgtr(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case dorgtr: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &tau))
            ){
                LAPACKE_dorgtr(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case sormbr: {
            int matrix_layout; char vect; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormbr(matrix_layout, vect, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case dormbr: {
            int matrix_layout; char vect; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormbr(matrix_layout, vect, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case sormhr: {
            int matrix_layout; char side; char trans; int m; int n; int ilo; int ihi; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &ilo, &ihi, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormhr(matrix_layout, side, trans, m, n, ilo, ihi, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case dormhr: {
            int matrix_layout; char side; char trans; int m; int n; int ilo; int ihi; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &ilo, &ihi, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormhr(matrix_layout, side, trans, m, n, ilo, ihi, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case sormlq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormlq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case dormlq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormlq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case sormql: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormql(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case dormql: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormql(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case sormqr: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormqr(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case dormqr: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormqr(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case sormrq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormrq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case dormrq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormrq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case sormrz: {
            int matrix_layout; char side; char trans; int m; int n; int k; int l; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &l, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormrz(matrix_layout, side, trans, m, n, k, l, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case dormrz: {
            int matrix_layout; char side; char trans; int m; int n; int k; int l; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &l, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormrz(matrix_layout, side, trans, m, n, k, l, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case sormtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_sormtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case dormtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_dormtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case spbcon: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &anorm, &rcond))
            ){
                LAPACKE_spbcon(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case dpbcon: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &anorm, &rcond))
            ){
                LAPACKE_dpbcon(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case cpbcon: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &anorm, &rcond))
            ){
                LAPACKE_cpbcon(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zpbcon: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &anorm, &rcond))
            ){
                LAPACKE_zpbcon(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case spbequ: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &s, &scond, &amax))
            ){
                LAPACKE_spbequ(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case dpbequ: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &s, &scond, &amax))
            ){
                LAPACKE_dpbequ(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case cpbequ: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &s, &scond, &amax))
            ){
                LAPACKE_cpbequ(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case zpbequ: {
            int matrix_layout; char uplo; int n; int kd; cste_c_binary ab; int ldab; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab, &s, &scond, &amax))
            ){
                LAPACKE_zpbequ(matrix_layout, uplo, n, kd, get_cste_ptr(ab), ldab, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case spbrfs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &afb, &ldafb, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_spbrfs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dpbrfs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &afb, &ldafb, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dpbrfs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cpbrfs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &afb, &ldafb, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cpbrfs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zpbrfs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary afb; int ldafb; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &afb, &ldafb, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zpbrfs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(afb), ldafb, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case spbstf: {
            int matrix_layout; char uplo; int n; int kb; c_binary bb; int ldbb;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kb, &bb, &ldbb))
            ){
                LAPACKE_spbstf(matrix_layout, uplo, n, kb, get_ptr(bb), ldbb);
            }
        break; }
        case dpbstf: {
            int matrix_layout; char uplo; int n; int kb; c_binary bb; int ldbb;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kb, &bb, &ldbb))
            ){
                LAPACKE_dpbstf(matrix_layout, uplo, n, kb, get_ptr(bb), ldbb);
            }
        break; }
        case cpbstf: {
            int matrix_layout; char uplo; int n; int kb; c_binary bb; int ldbb;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kb, &bb, &ldbb))
            ){
                LAPACKE_cpbstf(matrix_layout, uplo, n, kb, get_ptr(bb), ldbb);
            }
        break; }
        case zpbstf: {
            int matrix_layout; char uplo; int n; int kb; c_binary bb; int ldbb;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kb, &bb, &ldbb))
            ){
                LAPACKE_zpbstf(matrix_layout, uplo, n, kb, get_ptr(bb), ldbb);
            }
        break; }
        case spbsv: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_spbsv(matrix_layout, uplo, n, kd, nrhs, get_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case dpbsv: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_dpbsv(matrix_layout, uplo, n, kd, nrhs, get_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case cpbsv: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_cpbsv(matrix_layout, uplo, n, kd, nrhs, get_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case zpbsv: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_zpbsv(matrix_layout, uplo, n, kd, nrhs, get_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case spbtrf: {
            int matrix_layout; char uplo; int n; int kd; c_binary ab; int ldab;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab))
            ){
                LAPACKE_spbtrf(matrix_layout, uplo, n, kd, get_ptr(ab), ldab);
            }
        break; }
        case dpbtrf: {
            int matrix_layout; char uplo; int n; int kd; c_binary ab; int ldab;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab))
            ){
                LAPACKE_dpbtrf(matrix_layout, uplo, n, kd, get_ptr(ab), ldab);
            }
        break; }
        case cpbtrf: {
            int matrix_layout; char uplo; int n; int kd; c_binary ab; int ldab;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab))
            ){
                LAPACKE_cpbtrf(matrix_layout, uplo, n, kd, get_ptr(ab), ldab);
            }
        break; }
        case zpbtrf: {
            int matrix_layout; char uplo; int n; int kd; c_binary ab; int ldab;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &ab, &ldab))
            ){
                LAPACKE_zpbtrf(matrix_layout, uplo, n, kd, get_ptr(ab), ldab);
            }
        break; }
        case spbtrs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_spbtrs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case dpbtrs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_dpbtrs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case cpbtrs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_cpbtrs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case zpbtrs: {
            int matrix_layout; char uplo; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_zpbtrs(matrix_layout, uplo, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case spftrf: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_spftrf(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        break; }
        case dpftrf: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_dpftrf(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        break; }
        case cpftrf: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_cpftrf(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        break; }
        case zpftrf: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_zpftrf(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        break; }
        case spftri: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_spftri(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        break; }
        case dpftri: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_dpftri(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        break; }
        case cpftri: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_cpftri(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        break; }
        case zpftri: {
            int matrix_layout; char transr; char uplo; int n; c_binary a;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a))
            ){
                LAPACKE_zpftri(matrix_layout, transr, uplo, n, get_ptr(a));
            }
        break; }
        case spftrs: {
            int matrix_layout; char transr; char uplo; int n; int nrhs; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &nrhs, &a, &b, &ldb))
            ){
                LAPACKE_spftrs(matrix_layout, transr, uplo, n, nrhs, get_cste_ptr(a), get_ptr(b), ldb);
            }
        break; }
        case dpftrs: {
            int matrix_layout; char transr; char uplo; int n; int nrhs; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &nrhs, &a, &b, &ldb))
            ){
                LAPACKE_dpftrs(matrix_layout, transr, uplo, n, nrhs, get_cste_ptr(a), get_ptr(b), ldb);
            }
        break; }
        case cpftrs: {
            int matrix_layout; char transr; char uplo; int n; int nrhs; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &nrhs, &a, &b, &ldb))
            ){
                LAPACKE_cpftrs(matrix_layout, transr, uplo, n, nrhs, get_cste_ptr(a), get_ptr(b), ldb);
            }
        break; }
        case zpftrs: {
            int matrix_layout; char transr; char uplo; int n; int nrhs; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &nrhs, &a, &b, &ldb))
            ){
                LAPACKE_zpftrs(matrix_layout, transr, uplo, n, nrhs, get_cste_ptr(a), get_ptr(b), ldb);
            }
        break; }
        case spocon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_spocon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case dpocon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_dpocon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case cpocon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_cpocon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zpocon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &anorm, &rcond))
            ){
                LAPACKE_zpocon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case spoequ: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_spoequ(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case dpoequ: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_dpoequ(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case cpoequ: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_cpoequ(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case zpoequ: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_zpoequ(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case spoequb: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_spoequb(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case dpoequb: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_dpoequb(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case cpoequb: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_cpoequb(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case zpoequb: {
            int matrix_layout; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_zpoequb(matrix_layout, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case sporfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sporfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dporfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dporfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cporfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cporfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zporfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zporfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case sposv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_sposv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case dposv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dposv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case cposv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_cposv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case zposv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_zposv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case spotrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_spotrf(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        break; }
        case dpotrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_dpotrf(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        break; }
        case cpotrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_cpotrf(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        break; }
        case zpotrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_zpotrf(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        break; }
        case spotri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_spotri(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        break; }
        case dpotri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_dpotri(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        break; }
        case cpotri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_cpotri(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        break; }
        case zpotri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &a, &lda))
            ){
                LAPACKE_zpotri(matrix_layout, uplo, n, get_ptr(a), lda);
            }
        break; }
        case spotrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_spotrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case dpotrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dpotrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case cpotrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_cpotrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case zpotrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_zpotrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case sppcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &anorm, &rcond))
            ){
                LAPACKE_sppcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case dppcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &anorm, &rcond))
            ){
                LAPACKE_dppcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case cppcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &anorm, &rcond))
            ){
                LAPACKE_cppcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zppcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &anorm, &rcond))
            ){
                LAPACKE_zppcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case sppequ: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &s, &scond, &amax))
            ){
                LAPACKE_sppequ(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case dppequ: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &s, &scond, &amax))
            ){
                LAPACKE_dppequ(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case cppequ: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &s, &scond, &amax))
            ){
                LAPACKE_cppequ(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case zppequ: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &s, &scond, &amax))
            ){
                LAPACKE_zppequ(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case spprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_spprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dpprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dpprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cpprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cpprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zpprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zpprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case sppsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_sppsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case dppsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_dppsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case cppsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_cppsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case zppsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_zppsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case spptrf: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_spptrf(matrix_layout, uplo, n, get_ptr(ap));
            }
        break; }
        case dpptrf: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_dpptrf(matrix_layout, uplo, n, get_ptr(ap));
            }
        break; }
        case cpptrf: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_cpptrf(matrix_layout, uplo, n, get_ptr(ap));
            }
        break; }
        case zpptrf: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_zpptrf(matrix_layout, uplo, n, get_ptr(ap));
            }
        break; }
        case spptri: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_spptri(matrix_layout, uplo, n, get_ptr(ap));
            }
        break; }
        case dpptri: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_dpptri(matrix_layout, uplo, n, get_ptr(ap));
            }
        break; }
        case cpptri: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_cpptri(matrix_layout, uplo, n, get_ptr(ap));
            }
        break; }
        case zpptri: {
            int matrix_layout; char uplo; int n; c_binary ap;
            
            if( !(error = narg == 4? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap))
            ){
                LAPACKE_zpptri(matrix_layout, uplo, n, get_ptr(ap));
            }
        break; }
        case spptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_spptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case dpptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_dpptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case cpptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_cpptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case zpptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_zpptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case spstrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary piv; c_binary rank; cste_c_binary tol;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &piv, &rank, &tol))
            ){
                LAPACKE_spstrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(piv), get_ptr(rank), get_cste_double(tol));
            }
        break; }
        case dpstrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary piv; c_binary rank; cste_c_binary tol;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &piv, &rank, &tol))
            ){
                LAPACKE_dpstrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(piv), get_ptr(rank), get_cste_double(tol));
            }
        break; }
        case cpstrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary piv; c_binary rank; cste_c_binary tol;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &piv, &rank, &tol))
            ){
                LAPACKE_cpstrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(piv), get_ptr(rank), get_cste_double(tol));
            }
        break; }
        case zpstrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary piv; c_binary rank; cste_c_binary tol;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &piv, &rank, &tol))
            ){
                LAPACKE_zpstrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(piv), get_ptr(rank), get_cste_double(tol));
            }
        break; }
        case sptcon: {
            int n; cste_c_binary d; cste_c_binary e; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &n, &d, &e, &anorm, &rcond))
            ){
                LAPACKE_sptcon(n, get_cste_ptr(d), get_cste_ptr(e), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case dptcon: {
            int n; cste_c_binary d; cste_c_binary e; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &n, &d, &e, &anorm, &rcond))
            ){
                LAPACKE_dptcon(n, get_cste_ptr(d), get_cste_ptr(e), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case cptcon: {
            int n; cste_c_binary d; cste_c_binary e; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &n, &d, &e, &anorm, &rcond))
            ){
                LAPACKE_cptcon(n, get_cste_ptr(d), get_cste_ptr(e), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zptcon: {
            int n; cste_c_binary d; cste_c_binary e; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &n, &d, &e, &anorm, &rcond))
            ){
                LAPACKE_zptcon(n, get_cste_ptr(d), get_cste_ptr(e), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case spteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_spteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case dpteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dpteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case cpteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_cpteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case zpteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_zpteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case sptrfs: {
            int matrix_layout; int n; int nrhs; cste_c_binary d; cste_c_binary e; cste_c_binary df; cste_c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_sptrfs(matrix_layout, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_cste_ptr(df), get_cste_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dptrfs: {
            int matrix_layout; int n; int nrhs; cste_c_binary d; cste_c_binary e; cste_c_binary df; cste_c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dptrfs(matrix_layout, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_cste_ptr(df), get_cste_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cptrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary d; cste_c_binary e; cste_c_binary df; cste_c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_cptrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_cste_ptr(df), get_cste_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zptrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary d; cste_c_binary e; cste_c_binary df; cste_c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zptrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_cste_ptr(df), get_cste_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case sptsv: {
            int matrix_layout; int n; int nrhs; c_binary d; c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_sptsv(matrix_layout, n, nrhs, get_ptr(d), get_ptr(e), get_ptr(b), ldb);
            }
        break; }
        case dptsv: {
            int matrix_layout; int n; int nrhs; c_binary d; c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_dptsv(matrix_layout, n, nrhs, get_ptr(d), get_ptr(e), get_ptr(b), ldb);
            }
        break; }
        case cptsv: {
            int matrix_layout; int n; int nrhs; c_binary d; c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_cptsv(matrix_layout, n, nrhs, get_ptr(d), get_ptr(e), get_ptr(b), ldb);
            }
        break; }
        case zptsv: {
            int matrix_layout; int n; int nrhs; c_binary d; c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_zptsv(matrix_layout, n, nrhs, get_ptr(d), get_ptr(e), get_ptr(b), ldb);
            }
        break; }
        case sptsvx: {
            int matrix_layout; char fact; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary df; c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_sptsvx(matrix_layout, fact, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(df), get_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dptsvx: {
            int matrix_layout; char fact; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary df; c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_dptsvx(matrix_layout, fact, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(df), get_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cptsvx: {
            int matrix_layout; char fact; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary df; c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_cptsvx(matrix_layout, fact, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(df), get_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zptsvx: {
            int matrix_layout; char fact; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary df; c_binary ef; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &n, &nrhs, &d, &e, &df, &ef, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zptsvx(matrix_layout, fact, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(df), get_ptr(ef), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case spttrf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_spttrf(n, get_ptr(d), get_ptr(e));
            }
        break; }
        case dpttrf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_dpttrf(n, get_ptr(d), get_ptr(e));
            }
        break; }
        case cpttrf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_cpttrf(n, get_ptr(d), get_ptr(e));
            }
        break; }
        case zpttrf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_zpttrf(n, get_ptr(d), get_ptr(e));
            }
        break; }
        case spttrs: {
            int matrix_layout; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_spttrs(matrix_layout, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(b), ldb);
            }
        break; }
        case dpttrs: {
            int matrix_layout; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_dpttrs(matrix_layout, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(b), ldb);
            }
        break; }
        case cpttrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_cpttrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(b), ldb);
            }
        break; }
        case zpttrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary d; cste_c_binary e; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &d, &e, &b, &ldb))
            ){
                LAPACKE_zpttrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(d), get_cste_ptr(e), get_ptr(b), ldb);
            }
        break; }
        case ssbev: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_ssbev(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case dsbev: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_dsbev(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case ssbevd: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_ssbevd(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case dsbevd: {
            int matrix_layout; char jobz; char uplo; int n; int kd; c_binary ab; int ldab; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &kd, &ab, &ldab, &w, &z, &ldz))
            ){
                LAPACKE_dsbevd(matrix_layout, jobz, uplo, n, kd, get_ptr(ab), ldab, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case ssbevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int kd; c_binary ab; int ldab; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &kd, &ab, &ldab, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_ssbevx(matrix_layout, jobz, range, uplo, n, kd, get_ptr(ab), ldab, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case dsbevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int kd; c_binary ab; int ldab; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &kd, &ab, &ldab, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dsbevx(matrix_layout, jobz, range, uplo, n, kd, get_ptr(ab), ldab, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case ssbgst: {
            int matrix_layout; char vect; char uplo; int n; int ka; int kb; c_binary ab; int ldab; cste_c_binary bb; int ldbb; c_binary x; int ldx;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &x, &ldx))
            ){
                LAPACKE_ssbgst(matrix_layout, vect, uplo, n, ka, kb, get_ptr(ab), ldab, get_cste_ptr(bb), ldbb, get_ptr(x), ldx);
            }
        break; }
        case dsbgst: {
            int matrix_layout; char vect; char uplo; int n; int ka; int kb; c_binary ab; int ldab; cste_c_binary bb; int ldbb; c_binary x; int ldx;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &x, &ldx))
            ){
                LAPACKE_dsbgst(matrix_layout, vect, uplo, n, ka, kb, get_ptr(ab), ldab, get_cste_ptr(bb), ldbb, get_ptr(x), ldx);
            }
        break; }
        case ssbgv: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_ssbgv(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case dsbgv: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_dsbgv(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case ssbgvd: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_ssbgvd(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case dsbgvd: {
            int matrix_layout; char jobz; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &w, &z, &ldz))
            ){
                LAPACKE_dsbgvd(matrix_layout, jobz, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case ssbgvx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_ssbgvx(matrix_layout, jobz, range, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case dsbgvx: {
            int matrix_layout; char jobz; char range; char uplo; int n; int ka; int kb; c_binary ab; int ldab; c_binary bb; int ldbb; c_binary q; int ldq; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 23? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ka, &kb, &ab, &ldab, &bb, &ldbb, &q, &ldq, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dsbgvx(matrix_layout, jobz, range, uplo, n, ka, kb, get_ptr(ab), ldab, get_ptr(bb), ldbb, get_ptr(q), ldq, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case ssbtrd: {
            int matrix_layout; char vect; char uplo; int n; int kd; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &kd, &ab, &ldab, &d, &e, &q, &ldq))
            ){
                LAPACKE_ssbtrd(matrix_layout, vect, uplo, n, kd, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq);
            }
        break; }
        case dsbtrd: {
            int matrix_layout; char vect; char uplo; int n; int kd; c_binary ab; int ldab; c_binary d; c_binary e; c_binary q; int ldq;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &uplo, &n, &kd, &ab, &ldab, &d, &e, &q, &ldq))
            ){
                LAPACKE_dsbtrd(matrix_layout, vect, uplo, n, kd, get_ptr(ab), ldab, get_ptr(d), get_ptr(e), get_ptr(q), ldq);
            }
        break; }
        case ssfrk: {
            int matrix_layout; char transr; char uplo; char trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c))
            ){
                LAPACKE_ssfrk(matrix_layout, transr, uplo, trans, n, k, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_double(beta), get_ptr(c));
            }
        break; }
        case dsfrk: {
            int matrix_layout; char transr; char uplo; char trans; int n; int k; cste_c_binary alpha; cste_c_binary a; int lda; cste_c_binary beta; c_binary c;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &trans, &n, &k, &alpha, &a, &lda, &beta, &c))
            ){
                LAPACKE_dsfrk(matrix_layout, transr, uplo, trans, n, k, get_cste_double(alpha), get_cste_ptr(a), lda, get_cste_double(beta), get_ptr(c));
            }
        break; }
        case sspcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_sspcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case dspcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_dspcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case cspcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_cspcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zspcon: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zspcon(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case sspev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_sspev(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case dspev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_dspev(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case sspevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_sspevd(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case dspevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary ap; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &uplo, &n, &ap, &w, &z, &ldz))
            ){
                LAPACKE_dspevd(matrix_layout, jobz, uplo, n, get_ptr(ap), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case sspevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary ap; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ap, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_sspevx(matrix_layout, jobz, range, uplo, n, get_ptr(ap), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case dspevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary ap; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &ap, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dspevx(matrix_layout, jobz, range, uplo, n, get_ptr(ap), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case sspgst: {
            int matrix_layout; int itype; char uplo; int n; c_binary ap; cste_c_binary bp;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &itype, &uplo, &n, &ap, &bp))
            ){
                LAPACKE_sspgst(matrix_layout, itype, uplo, n, get_ptr(ap), get_cste_ptr(bp));
            }
        break; }
        case dspgst: {
            int matrix_layout; int itype; char uplo; int n; c_binary ap; cste_c_binary bp;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &itype, &uplo, &n, &ap, &bp))
            ){
                LAPACKE_dspgst(matrix_layout, itype, uplo, n, get_ptr(ap), get_cste_ptr(bp));
            }
        break; }
        case sspgv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_sspgv(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case dspgv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_dspgv(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case sspgvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_sspgvd(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case dspgvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary ap; c_binary bp; c_binary w; c_binary z; int ldz;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &ap, &bp, &w, &z, &ldz))
            ){
                LAPACKE_dspgvd(matrix_layout, itype, jobz, uplo, n, get_ptr(ap), get_ptr(bp), get_ptr(w), get_ptr(z), ldz);
            }
        break; }
        case sspgvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary ap; c_binary bp; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &ap, &bp, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_sspgvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(ap), get_ptr(bp), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case dspgvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary ap; c_binary bp; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 18? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &ap, &bp, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dspgvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(ap), get_ptr(bp), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case ssprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ssprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dsprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dsprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case csprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_csprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zsprfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary afp; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zsprfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(afp), get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case sspsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_sspsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case dspsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_dspsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case cspsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_cspsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zspsv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary ap; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_zspsv(matrix_layout, uplo, n, nrhs, get_ptr(ap), get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case sspsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_sspsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dspsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_dspsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case cspsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_cspsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zspsvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary ap; c_binary afp; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &ap, &afp, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zspsvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(ap), get_ptr(afp), get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case ssptrd: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &d, &e, &tau))
            ){
                LAPACKE_ssptrd(matrix_layout, uplo, n, get_ptr(ap), get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        break; }
        case dsptrd: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &d, &e, &tau))
            ){
                LAPACKE_dsptrd(matrix_layout, uplo, n, get_ptr(ap), get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        break; }
        case ssptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_ssptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        break; }
        case dsptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_dsptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        break; }
        case csptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_csptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        break; }
        case zsptrf: {
            int matrix_layout; char uplo; int n; c_binary ap; c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_zsptrf(matrix_layout, uplo, n, get_ptr(ap), get_ptr(ipiv));
            }
        break; }
        case ssptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_ssptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        break; }
        case dsptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_dsptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        break; }
        case csptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_csptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        break; }
        case zsptri: {
            int matrix_layout; char uplo; int n; c_binary ap; cste_c_binary ipiv;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &ap, &ipiv))
            ){
                LAPACKE_zsptri(matrix_layout, uplo, n, get_ptr(ap), get_cste_ptr(ipiv));
            }
        break; }
        case ssptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_ssptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case dsptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_dsptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case csptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_csptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zsptrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary ap; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &ap, &ipiv, &b, &ldb))
            ){
                LAPACKE_zsptrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(ap), get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case sstebz: {
            char range; char order; int n; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; cste_c_binary d; cste_c_binary e; c_binary m; c_binary nsplit; c_binary w; c_binary iblock; c_binary isplit;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_char, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &range, &order, &n, &vl, &vu, &il, &iu, &abstol, &d, &e, &m, &nsplit, &w, &iblock, &isplit))
            ){
                LAPACKE_sstebz(range, order, n, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_cste_ptr(d), get_cste_ptr(e), get_ptr(m), get_ptr(nsplit), get_ptr(w), get_ptr(iblock), get_ptr(isplit));
            }
        break; }
        case dstebz: {
            char range; char order; int n; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; cste_c_binary d; cste_c_binary e; c_binary m; c_binary nsplit; c_binary w; c_binary iblock; c_binary isplit;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_char, e_char, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &range, &order, &n, &vl, &vu, &il, &iu, &abstol, &d, &e, &m, &nsplit, &w, &iblock, &isplit))
            ){
                LAPACKE_dstebz(range, order, n, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_cste_ptr(d), get_cste_ptr(e), get_ptr(m), get_ptr(nsplit), get_ptr(w), get_ptr(iblock), get_ptr(isplit));
            }
        break; }
        case sstedc: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_sstedc(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case dstedc: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dstedc(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case cstedc: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_cstedc(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case zstedc: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_zstedc(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case sstegr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_sstegr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case dstegr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_dstegr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case cstegr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_cstegr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case zstegr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_zstegr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case sstein: {
            int matrix_layout; int n; cste_c_binary d; cste_c_binary e; int m; cste_c_binary w; cste_c_binary iblock; cste_c_binary isplit; c_binary z; int ldz; c_binary ifailv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &d, &e, &m, &w, &iblock, &isplit, &z, &ldz, &ifailv))
            ){
                LAPACKE_sstein(matrix_layout, n, get_cste_ptr(d), get_cste_ptr(e), m, get_cste_ptr(w), get_cste_ptr(iblock), get_cste_ptr(isplit), get_ptr(z), ldz, get_ptr(ifailv));
            }
        break; }
        case dstein: {
            int matrix_layout; int n; cste_c_binary d; cste_c_binary e; int m; cste_c_binary w; cste_c_binary iblock; cste_c_binary isplit; c_binary z; int ldz; c_binary ifailv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &d, &e, &m, &w, &iblock, &isplit, &z, &ldz, &ifailv))
            ){
                LAPACKE_dstein(matrix_layout, n, get_cste_ptr(d), get_cste_ptr(e), m, get_cste_ptr(w), get_cste_ptr(iblock), get_cste_ptr(isplit), get_ptr(z), ldz, get_ptr(ifailv));
            }
        break; }
        case cstein: {
            int matrix_layout; int n; cste_c_binary d; cste_c_binary e; int m; cste_c_binary w; cste_c_binary iblock; cste_c_binary isplit; c_binary z; int ldz; c_binary ifailv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &d, &e, &m, &w, &iblock, &isplit, &z, &ldz, &ifailv))
            ){
                LAPACKE_cstein(matrix_layout, n, get_cste_ptr(d), get_cste_ptr(e), m, get_cste_ptr(w), get_cste_ptr(iblock), get_cste_ptr(isplit), get_ptr(z), ldz, get_ptr(ifailv));
            }
        break; }
        case zstein: {
            int matrix_layout; int n; cste_c_binary d; cste_c_binary e; int m; cste_c_binary w; cste_c_binary iblock; cste_c_binary isplit; c_binary z; int ldz; c_binary ifailv;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &n, &d, &e, &m, &w, &iblock, &isplit, &z, &ldz, &ifailv))
            ){
                LAPACKE_zstein(matrix_layout, n, get_cste_ptr(d), get_cste_ptr(e), m, get_cste_ptr(w), get_cste_ptr(iblock), get_cste_ptr(isplit), get_ptr(z), ldz, get_ptr(ifailv));
            }
        break; }
        case sstemr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; c_binary m; c_binary w; c_binary z; int ldz; int nzc; c_binary isuppz; c_binary tryrac;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &m, &w, &z, &ldz, &nzc, &isuppz, &tryrac))
            ){
                LAPACKE_sstemr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_ptr(m), get_ptr(w), get_ptr(z), ldz, nzc, get_ptr(isuppz), get_ptr(tryrac));
            }
        break; }
        case dstemr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; c_binary m; c_binary w; c_binary z; int ldz; int nzc; c_binary isuppz; c_binary tryrac;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &m, &w, &z, &ldz, &nzc, &isuppz, &tryrac))
            ){
                LAPACKE_dstemr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_ptr(m), get_ptr(w), get_ptr(z), ldz, nzc, get_ptr(isuppz), get_ptr(tryrac));
            }
        break; }
        case cstemr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; c_binary m; c_binary w; c_binary z; int ldz; int nzc; c_binary isuppz; c_binary tryrac;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &m, &w, &z, &ldz, &nzc, &isuppz, &tryrac))
            ){
                LAPACKE_cstemr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_ptr(m), get_ptr(w), get_ptr(z), ldz, nzc, get_ptr(isuppz), get_ptr(tryrac));
            }
        break; }
        case zstemr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; c_binary m; c_binary w; c_binary z; int ldz; int nzc; c_binary isuppz; c_binary tryrac;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_ptr, e_ptr, e_ptr, e_int, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &m, &w, &z, &ldz, &nzc, &isuppz, &tryrac))
            ){
                LAPACKE_zstemr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_ptr(m), get_ptr(w), get_ptr(z), ldz, nzc, get_ptr(isuppz), get_ptr(tryrac));
            }
        break; }
        case ssteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_ssteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case dsteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dsteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case csteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_csteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case zsteqr: {
            int matrix_layout; char compz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &compz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_zsteqr(matrix_layout, compz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case ssterf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_ssterf(n, get_ptr(d), get_ptr(e));
            }
        break; }
        case dsterf: {
            int n; c_binary d; c_binary e;
            
            if( !(error = narg == 3? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_int, e_ptr, e_ptr, e_end}, &n, &d, &e))
            ){
                LAPACKE_dsterf(n, get_ptr(d), get_ptr(e));
            }
        break; }
        case sstev: {
            int matrix_layout; char jobz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_sstev(matrix_layout, jobz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case dstev: {
            int matrix_layout; char jobz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dstev(matrix_layout, jobz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case sstevd: {
            int matrix_layout; char jobz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_sstevd(matrix_layout, jobz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case dstevd: {
            int matrix_layout; char jobz; int n; c_binary d; c_binary e; c_binary z; int ldz;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &jobz, &n, &d, &e, &z, &ldz))
            ){
                LAPACKE_dstevd(matrix_layout, jobz, n, get_ptr(d), get_ptr(e), get_ptr(z), ldz);
            }
        break; }
        case sstevr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_sstevr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case dstevr: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_dstevr(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case sstevx: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_sstevx(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case dstevx: {
            int matrix_layout; char jobz; char range; int n; c_binary d; c_binary e; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 16? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_ptr, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &n, &d, &e, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dstevx(matrix_layout, jobz, range, n, get_ptr(d), get_ptr(e), get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case ssycon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_ssycon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case dsycon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_dsycon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case csycon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_csycon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case zsycon: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; cste_c_binary ipiv; cste_c_binary anorm; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv, &anorm, &rcond))
            ){
                LAPACKE_zsycon(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_cste_double(anorm), get_ptr(rcond));
            }
        break; }
        case ssyequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_ssyequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case dsyequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_dsyequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case csyequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_csyequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case zsyequb: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary s; c_binary scond; c_binary amax;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &s, &scond, &amax))
            ){
                LAPACKE_zsyequb(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(s), get_ptr(scond), get_ptr(amax));
            }
        break; }
        case ssyev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_ssyev(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        break; }
        case dsyev: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_dsyev(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        break; }
        case ssyevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_ssyevd(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        break; }
        case dsyevd: {
            int matrix_layout; char jobz; char uplo; int n; c_binary a; int lda; c_binary w;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &uplo, &n, &a, &lda, &w))
            ){
                LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, get_ptr(a), lda, get_ptr(w));
            }
        break; }
        case ssyevr: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_ssyevr(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case dsyevr: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary isuppz;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &isuppz))
            ){
                LAPACKE_dsyevr(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(isuppz));
            }
        break; }
        case ssyevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_ssyevx(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case dsyevx: {
            int matrix_layout; char jobz; char range; char uplo; int n; c_binary a; int lda; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobz, &range, &uplo, &n, &a, &lda, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dsyevx(matrix_layout, jobz, range, uplo, n, get_ptr(a), lda, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case ssygst: {
            int matrix_layout; int itype; char uplo; int n; c_binary a; int lda; cste_c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_end}, &matrix_layout, &itype, &uplo, &n, &a, &lda, &b, &ldb))
            ){
                LAPACKE_ssygst(matrix_layout, itype, uplo, n, get_ptr(a), lda, get_cste_ptr(b), ldb);
            }
        break; }
        case dsygst: {
            int matrix_layout; int itype; char uplo; int n; c_binary a; int lda; cste_c_binary b; int ldb;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_end}, &matrix_layout, &itype, &uplo, &n, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dsygst(matrix_layout, itype, uplo, n, get_ptr(a), lda, get_cste_ptr(b), ldb);
            }
        break; }
        case ssygv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_ssygv(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        break; }
        case dsygv: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_dsygv(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        break; }
        case ssygvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_ssygvd(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        break; }
        case dsygvd: {
            int matrix_layout; int itype; char jobz; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; c_binary w;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &uplo, &n, &a, &lda, &b, &ldb, &w))
            ){
                LAPACKE_dsygvd(matrix_layout, itype, jobz, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(w));
            }
        break; }
        case ssygvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &a, &lda, &b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_ssygvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case dsygvx: {
            int matrix_layout; int itype; char jobz; char range; char uplo; int n; c_binary a; int lda; c_binary b; int ldb; cste_c_binary vl; cste_c_binary vu; int il; int iu; cste_c_binary abstol; c_binary m; c_binary w; c_binary z; int ldz; c_binary ifail;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_char, e_char, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_int, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &itype, &jobz, &range, &uplo, &n, &a, &lda, &b, &ldb, &vl, &vu, &il, &iu, &abstol, &m, &w, &z, &ldz, &ifail))
            ){
                LAPACKE_dsygvx(matrix_layout, itype, jobz, range, uplo, n, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(vl), get_cste_double(vu), il, iu, get_cste_double(abstol), get_ptr(m), get_ptr(w), get_ptr(z), ldz, get_ptr(ifail));
            }
        break; }
        case ssyrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ssyrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dsyrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dsyrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case csyrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_csyrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zsyrfs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary af; int ldaf; cste_c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_zsyrfs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(af), ldaf, get_cste_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case ssysv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_ssysv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case dsysv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_dsysv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case csysv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_csysv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zsysv: {
            int matrix_layout; char uplo; int n; int nrhs; c_binary a; int lda; c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_ptr, e_int, e_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zsysv(matrix_layout, uplo, n, nrhs, get_ptr(a), lda, get_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case ssysvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_ssysvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dsysvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_dsysvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case csysvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_csysvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case zsysvx: {
            int matrix_layout; char fact; char uplo; int n; int nrhs; cste_c_binary a; int lda; c_binary af; int ldaf; c_binary ipiv; cste_c_binary b; int ldb; c_binary x; int ldx; c_binary rcond; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &fact, &uplo, &n, &nrhs, &a, &lda, &af, &ldaf, &ipiv, &b, &ldb, &x, &ldx, &rcond, &ferr, &berr))
            ){
                LAPACKE_zsysvx(matrix_layout, fact, uplo, n, nrhs, get_cste_ptr(a), lda, get_ptr(af), ldaf, get_ptr(ipiv), get_cste_ptr(b), ldb, get_ptr(x), ldx, get_ptr(rcond), get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case ssytrd: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &d, &e, &tau))
            ){
                LAPACKE_ssytrd(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        break; }
        case dsytrd: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary d; c_binary e; c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &d, &e, &tau))
            ){
                LAPACKE_dsytrd(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(d), get_ptr(e), get_ptr(tau));
            }
        break; }
        case ssytrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_ssytrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case dsytrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_dsytrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case csytrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_csytrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case zsytrf: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zsytrf(matrix_layout, uplo, n, get_ptr(a), lda, get_ptr(ipiv));
            }
        break; }
        case ssytri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_ssytri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case dsytri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_dsytri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case csytri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_csytri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case zsytri: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary ipiv;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ipiv))
            ){
                LAPACKE_zsytri(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(ipiv));
            }
        break; }
        case ssytrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_ssytrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case dsytrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_dsytrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case csytrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_csytrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case zsytrs: {
            int matrix_layout; char uplo; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary ipiv; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &nrhs, &a, &lda, &ipiv, &b, &ldb))
            ){
                LAPACKE_zsytrs(matrix_layout, uplo, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(ipiv), get_ptr(b), ldb);
            }
        break; }
        case stbcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; int kd; cste_c_binary ab; int ldab; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &kd, &ab, &ldab, &rcond))
            ){
                LAPACKE_stbcon(matrix_layout, norm, uplo, diag, n, kd, get_cste_ptr(ab), ldab, get_ptr(rcond));
            }
        break; }
        case dtbcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; int kd; cste_c_binary ab; int ldab; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &kd, &ab, &ldab, &rcond))
            ){
                LAPACKE_dtbcon(matrix_layout, norm, uplo, diag, n, kd, get_cste_ptr(ab), ldab, get_ptr(rcond));
            }
        break; }
        case ctbcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; int kd; cste_c_binary ab; int ldab; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &kd, &ab, &ldab, &rcond))
            ){
                LAPACKE_ctbcon(matrix_layout, norm, uplo, diag, n, kd, get_cste_ptr(ab), ldab, get_ptr(rcond));
            }
        break; }
        case ztbcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; int kd; cste_c_binary ab; int ldab; c_binary rcond;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &kd, &ab, &ldab, &rcond))
            ){
                LAPACKE_ztbcon(matrix_layout, norm, uplo, diag, n, kd, get_cste_ptr(ab), ldab, get_ptr(rcond));
            }
        break; }
        case stbrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_stbrfs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dtbrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dtbrfs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case ctbrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ctbrfs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case ztbrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ztbrfs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case stbtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_stbtrs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case dtbtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_dtbtrs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case ctbtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_ctbtrs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case ztbtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int kd; int nrhs; cste_c_binary ab; int ldab; c_binary b; int ldb;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &kd, &nrhs, &ab, &ldab, &b, &ldb))
            ){
                LAPACKE_ztbtrs(matrix_layout, uplo, trans, diag, n, kd, nrhs, get_cste_ptr(ab), ldab, get_ptr(b), ldb);
            }
        break; }
        case stfsm: {
            int matrix_layout; char transr; char side; char uplo; char trans; char diag; int m; int n; cste_c_binary alpha; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &side, &uplo, &trans, &diag, &m, &n, &alpha, &a, &b, &ldb))
            ){
                LAPACKE_stfsm(matrix_layout, transr, side, uplo, trans, diag, m, n, get_cste_double(alpha), get_cste_ptr(a), get_ptr(b), ldb);
            }
        break; }
        case dtfsm: {
            int matrix_layout; char transr; char side; char uplo; char trans; char diag; int m; int n; cste_c_binary alpha; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &side, &uplo, &trans, &diag, &m, &n, &alpha, &a, &b, &ldb))
            ){
                LAPACKE_dtfsm(matrix_layout, transr, side, uplo, trans, diag, m, n, get_cste_double(alpha), get_cste_ptr(a), get_ptr(b), ldb);
            }
        break; }
        case ctfsm: {
            int matrix_layout; char transr; char side; char uplo; char trans; char diag; int m; int n; c_binary alpha; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &side, &uplo, &trans, &diag, &m, &n, &alpha, &a, &b, &ldb))
            ){
                LAPACKE_ctfsm(matrix_layout, transr, side, uplo, trans, diag, m, n, lapack_make_complex_float(*(float*)get_ptr(alpha), *(((float*)get_ptr(alpha))+1)), get_cste_ptr(a), get_ptr(b), ldb);
            }
        break; }
        case ztfsm: {
            int matrix_layout; char transr; char side; char uplo; char trans; char diag; int m; int n; c_binary alpha; cste_c_binary a; c_binary b; int ldb;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_char, e_char, e_int, e_int, e_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &side, &uplo, &trans, &diag, &m, &n, &alpha, &a, &b, &ldb))
            ){
                LAPACKE_ztfsm(matrix_layout, transr, side, uplo, trans, diag, m, n, lapack_make_complex_double(*(double*)get_ptr(alpha), *(((double*)get_ptr(alpha))+1)), get_cste_ptr(a), get_ptr(b), ldb);
            }
        break; }
        case stftri: {
            int matrix_layout; char transr; char uplo; char diag; int n; c_binary a;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &diag, &n, &a))
            ){
                LAPACKE_stftri(matrix_layout, transr, uplo, diag, n, get_ptr(a));
            }
        break; }
        case dtftri: {
            int matrix_layout; char transr; char uplo; char diag; int n; c_binary a;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &diag, &n, &a))
            ){
                LAPACKE_dtftri(matrix_layout, transr, uplo, diag, n, get_ptr(a));
            }
        break; }
        case ctftri: {
            int matrix_layout; char transr; char uplo; char diag; int n; c_binary a;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &diag, &n, &a))
            ){
                LAPACKE_ctftri(matrix_layout, transr, uplo, diag, n, get_ptr(a));
            }
        break; }
        case ztftri: {
            int matrix_layout; char transr; char uplo; char diag; int n; c_binary a;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &diag, &n, &a))
            ){
                LAPACKE_ztftri(matrix_layout, transr, uplo, diag, n, get_ptr(a));
            }
        break; }
        case stfttp: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &ap))
            ){
                LAPACKE_stfttp(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(ap));
            }
        break; }
        case dtfttp: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &ap))
            ){
                LAPACKE_dtfttp(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(ap));
            }
        break; }
        case ctfttp: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &ap))
            ){
                LAPACKE_ctfttp(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(ap));
            }
        break; }
        case ztfttp: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &ap))
            ){
                LAPACKE_ztfttp(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(ap));
            }
        break; }
        case stfttr: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary a; int lda;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &a, &lda))
            ){
                LAPACKE_stfttr(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(a), lda);
            }
        break; }
        case dtfttr: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary a; int lda;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &a, &lda))
            ){
                LAPACKE_dtfttr(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(a), lda);
            }
        break; }
        case ctfttr: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary a; int lda;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &a, &lda))
            ){
                LAPACKE_ctfttr(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(a), lda);
            }
        break; }
        case ztfttr: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary arf; c_binary a; int lda;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &transr, &uplo, &n, &arf, &a, &lda))
            ){
                LAPACKE_ztfttr(matrix_layout, transr, uplo, n, get_cste_ptr(arf), get_ptr(a), lda);
            }
        break; }
        case stgevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; cste_c_binary s; int lds; cste_c_binary p; int ldp; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &s, &lds, &p, &ldp, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_stgevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_cste_ptr(s), lds, get_cste_ptr(p), ldp, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        break; }
        case dtgevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; cste_c_binary s; int lds; cste_c_binary p; int ldp; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &s, &lds, &p, &ldp, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_dtgevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_cste_ptr(s), lds, get_cste_ptr(p), ldp, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        break; }
        case ctgevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; cste_c_binary s; int lds; cste_c_binary p; int ldp; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &s, &lds, &p, &ldp, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_ctgevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_cste_ptr(s), lds, get_cste_ptr(p), ldp, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        break; }
        case ztgevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; cste_c_binary s; int lds; cste_c_binary p; int ldp; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &s, &lds, &p, &ldp, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_ztgevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_cste_ptr(s), lds, get_cste_ptr(p), ldp, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        break; }
        case stgexc: {
            int matrix_layout; int wantq; int wantz; int n; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz; c_binary ifst; c_binary ilst;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &wantq, &wantz, &n, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz, &ifst, &ilst))
            ){
                LAPACKE_stgexc(matrix_layout, wantq, wantz, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(ifst), get_ptr(ilst));
            }
        break; }
        case dtgexc: {
            int matrix_layout; int wantq; int wantz; int n; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz; c_binary ifst; c_binary ilst;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &wantq, &wantz, &n, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz, &ifst, &ilst))
            ){
                LAPACKE_dtgexc(matrix_layout, wantq, wantz, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(ifst), get_ptr(ilst));
            }
        break; }
        case ctgexc: {
            int matrix_layout; int wantq; int wantz; int n; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz; int ifst; int ilst;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_int, e_end}, &matrix_layout, &wantq, &wantz, &n, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz, &ifst, &ilst))
            ){
                LAPACKE_ctgexc(matrix_layout, wantq, wantz, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz, ifst, ilst);
            }
        break; }
        case ztgexc: {
            int matrix_layout; int wantq; int wantz; int n; c_binary a; int lda; c_binary b; int ldb; c_binary q; int ldq; c_binary z; int ldz; int ifst; int ilst;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_int, e_end}, &matrix_layout, &wantq, &wantz, &n, &a, &lda, &b, &ldb, &q, &ldq, &z, &ldz, &ifst, &ilst))
            ){
                LAPACKE_ztgexc(matrix_layout, wantq, wantz, n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(q), ldq, get_ptr(z), ldz, ifst, ilst);
            }
        break; }
        case stgsen: {
            int matrix_layout; int ijob; int wantq; int wantz; cste_c_binary select; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary q; int ldq; c_binary z; int ldz; c_binary m; c_binary pl; c_binary pr; c_binary dif;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &ijob, &wantq, &wantz, &select, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &q, &ldq, &z, &ldz, &m, &pl, &pr, &dif))
            ){
                LAPACKE_stgsen(matrix_layout, ijob, wantq, wantz, get_cste_ptr(select), n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(m), get_ptr(pl), get_ptr(pr), get_ptr(dif));
            }
        break; }
        case dtgsen: {
            int matrix_layout; int ijob; int wantq; int wantz; cste_c_binary select; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alphar; c_binary alphai; c_binary beta; c_binary q; int ldq; c_binary z; int ldz; c_binary m; c_binary pl; c_binary pr; c_binary dif;
            
            if( !(error = narg == 21? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &ijob, &wantq, &wantz, &select, &n, &a, &lda, &b, &ldb, &alphar, &alphai, &beta, &q, &ldq, &z, &ldz, &m, &pl, &pr, &dif))
            ){
                LAPACKE_dtgsen(matrix_layout, ijob, wantq, wantz, get_cste_ptr(select), n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alphar), get_ptr(alphai), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(m), get_ptr(pl), get_ptr(pr), get_ptr(dif));
            }
        break; }
        case ctgsen: {
            int matrix_layout; int ijob; int wantq; int wantz; cste_c_binary select; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary q; int ldq; c_binary z; int ldz; c_binary m; c_binary pl; c_binary pr; c_binary dif;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &ijob, &wantq, &wantz, &select, &n, &a, &lda, &b, &ldb, &alpha, &beta, &q, &ldq, &z, &ldz, &m, &pl, &pr, &dif))
            ){
                LAPACKE_ctgsen(matrix_layout, ijob, wantq, wantz, get_cste_ptr(select), n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(m), get_ptr(pl), get_ptr(pr), get_ptr(dif));
            }
        break; }
        case ztgsen: {
            int matrix_layout; int ijob; int wantq; int wantz; cste_c_binary select; int n; c_binary a; int lda; c_binary b; int ldb; c_binary alpha; c_binary beta; c_binary q; int ldq; c_binary z; int ldz; c_binary m; c_binary pl; c_binary pr; c_binary dif;
            
            if( !(error = narg == 20? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &ijob, &wantq, &wantz, &select, &n, &a, &lda, &b, &ldb, &alpha, &beta, &q, &ldq, &z, &ldz, &m, &pl, &pr, &dif))
            ){
                LAPACKE_ztgsen(matrix_layout, ijob, wantq, wantz, get_cste_ptr(select), n, get_ptr(a), lda, get_ptr(b), ldb, get_ptr(alpha), get_ptr(beta), get_ptr(q), ldq, get_ptr(z), ldz, get_ptr(m), get_ptr(pl), get_ptr(pr), get_ptr(dif));
            }
        break; }
        case stgsja: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; int k; int l; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary ncycle;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, &a, &lda, &b, &ldb, &tola, &tolb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &ncycle))
            ){
                LAPACKE_stgsja(matrix_layout, jobu, jobv, jobq, m, p, n, k, l, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(ncycle));
            }
        break; }
        case dtgsja: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; int k; int l; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary ncycle;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, &a, &lda, &b, &ldb, &tola, &tolb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &ncycle))
            ){
                LAPACKE_dtgsja(matrix_layout, jobu, jobv, jobq, m, p, n, k, l, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(ncycle));
            }
        break; }
        case ctgsja: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; int k; int l; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary ncycle;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, &a, &lda, &b, &ldb, &tola, &tolb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &ncycle))
            ){
                LAPACKE_ctgsja(matrix_layout, jobu, jobv, jobq, m, p, n, k, l, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(ncycle));
            }
        break; }
        case ztgsja: {
            int matrix_layout; char jobu; char jobv; char jobq; int m; int p; int n; int k; int l; c_binary a; int lda; c_binary b; int ldb; cste_c_binary tola; cste_c_binary tolb; c_binary alpha; c_binary beta; c_binary u; int ldu; c_binary v; int ldv; c_binary q; int ldq; c_binary ncycle;
            
            if( !(error = narg == 24? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_int, e_int, e_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_ptr, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &jobu, &jobv, &jobq, &m, &p, &n, &k, &l, &a, &lda, &b, &ldb, &tola, &tolb, &alpha, &beta, &u, &ldu, &v, &ldv, &q, &ldq, &ncycle))
            ){
                LAPACKE_ztgsja(matrix_layout, jobu, jobv, jobq, m, p, n, k, l, get_ptr(a), lda, get_ptr(b), ldb, get_cste_double(tola), get_cste_double(tolb), get_ptr(alpha), get_ptr(beta), get_ptr(u), ldu, get_ptr(v), ldv, get_ptr(q), ldq, get_ptr(ncycle));
            }
        break; }
        case stgsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary dif; int mm; c_binary m;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &a, &lda, &b, &ldb, &vl, &ldvl, &vr, &ldvr, &s, &dif, &mm, &m))
            ){
                LAPACKE_stgsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(dif), mm, get_ptr(m));
            }
        break; }
        case dtgsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary dif; int mm; c_binary m;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &a, &lda, &b, &ldb, &vl, &ldvl, &vr, &ldvr, &s, &dif, &mm, &m))
            ){
                LAPACKE_dtgsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(dif), mm, get_ptr(m));
            }
        break; }
        case ctgsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary dif; int mm; c_binary m;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &a, &lda, &b, &ldb, &vl, &ldvl, &vr, &ldvr, &s, &dif, &mm, &m))
            ){
                LAPACKE_ctgsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(dif), mm, get_ptr(m));
            }
        break; }
        case ztgsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary dif; int mm; c_binary m;
            
            if( !(error = narg == 17? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &a, &lda, &b, &ldb, &vl, &ldvl, &vr, &ldvr, &s, &dif, &mm, &m))
            ){
                LAPACKE_ztgsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(dif), mm, get_ptr(m));
            }
        break; }
        case stgsyl: {
            int matrix_layout; char trans; int ijob; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; cste_c_binary d; int ldd; cste_c_binary e; int lde; c_binary f; int ldf; c_binary scale; c_binary dif;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &ijob, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &d, &ldd, &e, &lde, &f, &ldf, &scale, &dif))
            ){
                LAPACKE_stgsyl(matrix_layout, trans, ijob, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_cste_ptr(d), ldd, get_cste_ptr(e), lde, get_ptr(f), ldf, get_ptr(scale), get_ptr(dif));
            }
        break; }
        case dtgsyl: {
            int matrix_layout; char trans; int ijob; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; cste_c_binary d; int ldd; cste_c_binary e; int lde; c_binary f; int ldf; c_binary scale; c_binary dif;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &ijob, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &d, &ldd, &e, &lde, &f, &ldf, &scale, &dif))
            ){
                LAPACKE_dtgsyl(matrix_layout, trans, ijob, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_cste_ptr(d), ldd, get_cste_ptr(e), lde, get_ptr(f), ldf, get_ptr(scale), get_ptr(dif));
            }
        break; }
        case ctgsyl: {
            int matrix_layout; char trans; int ijob; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; cste_c_binary d; int ldd; cste_c_binary e; int lde; c_binary f; int ldf; c_binary scale; c_binary dif;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &ijob, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &d, &ldd, &e, &lde, &f, &ldf, &scale, &dif))
            ){
                LAPACKE_ctgsyl(matrix_layout, trans, ijob, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_cste_ptr(d), ldd, get_cste_ptr(e), lde, get_ptr(f), ldf, get_ptr(scale), get_ptr(dif));
            }
        break; }
        case ztgsyl: {
            int matrix_layout; char trans; int ijob; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; cste_c_binary d; int ldd; cste_c_binary e; int lde; c_binary f; int ldf; c_binary scale; c_binary dif;
            
            if( !(error = narg == 19? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &trans, &ijob, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &d, &ldd, &e, &lde, &f, &ldf, &scale, &dif))
            ){
                LAPACKE_ztgsyl(matrix_layout, trans, ijob, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_cste_ptr(d), ldd, get_cste_ptr(e), lde, get_ptr(f), ldf, get_ptr(scale), get_ptr(dif));
            }
        break; }
        case stpcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary ap; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &ap, &rcond))
            ){
                LAPACKE_stpcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(ap), get_ptr(rcond));
            }
        break; }
        case dtpcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary ap; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &ap, &rcond))
            ){
                LAPACKE_dtpcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(ap), get_ptr(rcond));
            }
        break; }
        case ctpcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary ap; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &ap, &rcond))
            ){
                LAPACKE_ctpcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(ap), get_ptr(rcond));
            }
        break; }
        case ztpcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary ap; c_binary rcond;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &ap, &rcond))
            ){
                LAPACKE_ztpcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(ap), get_ptr(rcond));
            }
        break; }
        case stprfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_stprfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dtprfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dtprfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case ctprfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ctprfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case ztprfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ztprfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case stptri: {
            int matrix_layout; char uplo; char diag; int n; c_binary ap;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &diag, &n, &ap))
            ){
                LAPACKE_stptri(matrix_layout, uplo, diag, n, get_ptr(ap));
            }
        break; }
        case dtptri: {
            int matrix_layout; char uplo; char diag; int n; c_binary ap;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &diag, &n, &ap))
            ){
                LAPACKE_dtptri(matrix_layout, uplo, diag, n, get_ptr(ap));
            }
        break; }
        case ctptri: {
            int matrix_layout; char uplo; char diag; int n; c_binary ap;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &diag, &n, &ap))
            ){
                LAPACKE_ctptri(matrix_layout, uplo, diag, n, get_ptr(ap));
            }
        break; }
        case ztptri: {
            int matrix_layout; char uplo; char diag; int n; c_binary ap;
            
            if( !(error = narg == 5? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &diag, &n, &ap))
            ){
                LAPACKE_ztptri(matrix_layout, uplo, diag, n, get_ptr(ap));
            }
        break; }
        case stptrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_stptrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case dtptrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_dtptrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case ctptrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_ctptrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case ztptrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary ap; c_binary b; int ldb;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &ap, &b, &ldb))
            ){
                LAPACKE_ztptrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(ap), get_ptr(b), ldb);
            }
        break; }
        case stpttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary ap; c_binary arf;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &ap, &arf))
            ){
                LAPACKE_stpttf(matrix_layout, transr, uplo, n, get_cste_ptr(ap), get_ptr(arf));
            }
        break; }
        case dtpttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary ap; c_binary arf;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &ap, &arf))
            ){
                LAPACKE_dtpttf(matrix_layout, transr, uplo, n, get_cste_ptr(ap), get_ptr(arf));
            }
        break; }
        case ctpttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary ap; c_binary arf;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &ap, &arf))
            ){
                LAPACKE_ctpttf(matrix_layout, transr, uplo, n, get_cste_ptr(ap), get_ptr(arf));
            }
        break; }
        case ztpttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary ap; c_binary arf;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &ap, &arf))
            ){
                LAPACKE_ztpttf(matrix_layout, transr, uplo, n, get_cste_ptr(ap), get_ptr(arf));
            }
        break; }
        case stpttr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &a, &lda))
            ){
                LAPACKE_stpttr(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(a), lda);
            }
        break; }
        case dtpttr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &a, &lda))
            ){
                LAPACKE_dtpttr(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(a), lda);
            }
        break; }
        case ctpttr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &a, &lda))
            ){
                LAPACKE_ctpttr(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(a), lda);
            }
        break; }
        case ztpttr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &a, &lda))
            ){
                LAPACKE_ztpttr(matrix_layout, uplo, n, get_cste_ptr(ap), get_ptr(a), lda);
            }
        break; }
        case strcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary a; int lda; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &a, &lda, &rcond))
            ){
                LAPACKE_strcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(a), lda, get_ptr(rcond));
            }
        break; }
        case dtrcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary a; int lda; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &a, &lda, &rcond))
            ){
                LAPACKE_dtrcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(a), lda, get_ptr(rcond));
            }
        break; }
        case ctrcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary a; int lda; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &a, &lda, &rcond))
            ){
                LAPACKE_ctrcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(a), lda, get_ptr(rcond));
            }
        break; }
        case ztrcon: {
            int matrix_layout; char norm; char uplo; char diag; int n; cste_c_binary a; int lda; c_binary rcond;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &norm, &uplo, &diag, &n, &a, &lda, &rcond))
            ){
                LAPACKE_ztrcon(matrix_layout, norm, uplo, diag, n, get_cste_ptr(a), lda, get_ptr(rcond));
            }
        break; }
        case strevc: {
            int matrix_layout; char side; char howmny; c_binary select; int n; cste_c_binary t; int ldt; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_strevc(matrix_layout, side, howmny, get_ptr(select), n, get_cste_ptr(t), ldt, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        break; }
        case dtrevc: {
            int matrix_layout; char side; char howmny; c_binary select; int n; cste_c_binary t; int ldt; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_dtrevc(matrix_layout, side, howmny, get_ptr(select), n, get_cste_ptr(t), ldt, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        break; }
        case ctrevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; c_binary t; int ldt; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_ctrevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        break; }
        case ztrevc: {
            int matrix_layout; char side; char howmny; cste_c_binary select; int n; c_binary t; int ldt; c_binary vl; int ldvl; c_binary vr; int ldvr; int mm; c_binary m;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_ptr, e_end}, &matrix_layout, &side, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &mm, &m))
            ){
                LAPACKE_ztrevc(matrix_layout, side, howmny, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(vl), ldvl, get_ptr(vr), ldvr, mm, get_ptr(m));
            }
        break; }
        case strexc: {
            int matrix_layout; char compq; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary ifst; c_binary ilst;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &compq, &n, &t, &ldt, &q, &ldq, &ifst, &ilst))
            ){
                LAPACKE_strexc(matrix_layout, compq, n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(ifst), get_ptr(ilst));
            }
        break; }
        case dtrexc: {
            int matrix_layout; char compq; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary ifst; c_binary ilst;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &compq, &n, &t, &ldt, &q, &ldq, &ifst, &ilst))
            ){
                LAPACKE_dtrexc(matrix_layout, compq, n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(ifst), get_ptr(ilst));
            }
        break; }
        case ctrexc: {
            int matrix_layout; char compq; int n; c_binary t; int ldt; c_binary q; int ldq; int ifst; int ilst;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_int, e_end}, &matrix_layout, &compq, &n, &t, &ldt, &q, &ldq, &ifst, &ilst))
            ){
                LAPACKE_ctrexc(matrix_layout, compq, n, get_ptr(t), ldt, get_ptr(q), ldq, ifst, ilst);
            }
        break; }
        case ztrexc: {
            int matrix_layout; char compq; int n; c_binary t; int ldt; c_binary q; int ldq; int ifst; int ilst;
            
            if( !(error = narg == 9? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_ptr, e_int, e_int, e_int, e_end}, &matrix_layout, &compq, &n, &t, &ldt, &q, &ldq, &ifst, &ilst))
            ){
                LAPACKE_ztrexc(matrix_layout, compq, n, get_ptr(t), ldt, get_ptr(q), ldq, ifst, ilst);
            }
        break; }
        case strrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_strrfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case dtrrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_dtrrfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case ctrrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ctrrfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case ztrrfs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; cste_c_binary b; int ldb; cste_c_binary x; int ldx; c_binary ferr; c_binary berr;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb, &x, &ldx, &ferr, &berr))
            ){
                LAPACKE_ztrrfs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_cste_ptr(x), ldx, get_ptr(ferr), get_ptr(berr));
            }
        break; }
        case strsen: {
            int matrix_layout; char job; char compq; cste_c_binary select; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary wr; c_binary wi; c_binary m; c_binary s; c_binary sep;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &compq, &select, &n, &t, &ldt, &q, &ldq, &wr, &wi, &m, &s, &sep))
            ){
                LAPACKE_strsen(matrix_layout, job, compq, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(wr), get_ptr(wi), get_ptr(m), get_ptr(s), get_ptr(sep));
            }
        break; }
        case dtrsen: {
            int matrix_layout; char job; char compq; cste_c_binary select; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary wr; c_binary wi; c_binary m; c_binary s; c_binary sep;
            
            if( !(error = narg == 14? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &compq, &select, &n, &t, &ldt, &q, &ldq, &wr, &wi, &m, &s, &sep))
            ){
                LAPACKE_dtrsen(matrix_layout, job, compq, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(wr), get_ptr(wi), get_ptr(m), get_ptr(s), get_ptr(sep));
            }
        break; }
        case ctrsen: {
            int matrix_layout; char job; char compq; cste_c_binary select; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary w; c_binary m; c_binary s; c_binary sep;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &compq, &select, &n, &t, &ldt, &q, &ldq, &w, &m, &s, &sep))
            ){
                LAPACKE_ctrsen(matrix_layout, job, compq, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(w), get_ptr(m), get_ptr(s), get_ptr(sep));
            }
        break; }
        case ztrsen: {
            int matrix_layout; char job; char compq; cste_c_binary select; int n; c_binary t; int ldt; c_binary q; int ldq; c_binary w; c_binary m; c_binary s; c_binary sep;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_int, e_ptr, e_ptr, e_ptr, e_ptr, e_end}, &matrix_layout, &job, &compq, &select, &n, &t, &ldt, &q, &ldq, &w, &m, &s, &sep))
            ){
                LAPACKE_ztrsen(matrix_layout, job, compq, get_cste_ptr(select), n, get_ptr(t), ldt, get_ptr(q), ldq, get_ptr(w), get_ptr(m), get_ptr(s), get_ptr(sep));
            }
        break; }
        case strsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary t; int ldt; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary sep; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &s, &sep, &mm, &m))
            ){
                LAPACKE_strsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(t), ldt, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(sep), mm, get_ptr(m));
            }
        break; }
        case dtrsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary t; int ldt; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary sep; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &s, &sep, &mm, &m))
            ){
                LAPACKE_dtrsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(t), ldt, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(sep), mm, get_ptr(m));
            }
        break; }
        case ctrsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary t; int ldt; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary sep; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &s, &sep, &mm, &m))
            ){
                LAPACKE_ctrsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(t), ldt, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(sep), mm, get_ptr(m));
            }
        break; }
        case ztrsna: {
            int matrix_layout; char job; char howmny; cste_c_binary select; int n; cste_c_binary t; int ldt; cste_c_binary vl; int ldvl; cste_c_binary vr; int ldvr; c_binary s; c_binary sep; int mm; c_binary m;
            
            if( !(error = narg == 15? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &job, &howmny, &select, &n, &t, &ldt, &vl, &ldvl, &vr, &ldvr, &s, &sep, &mm, &m))
            ){
                LAPACKE_ztrsna(matrix_layout, job, howmny, get_cste_ptr(select), n, get_cste_ptr(t), ldt, get_cste_ptr(vl), ldvl, get_cste_ptr(vr), ldvr, get_ptr(s), get_ptr(sep), mm, get_ptr(m));
            }
        break; }
        case strsyl: {
            int matrix_layout; char trana; char tranb; int isgn; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; c_binary scale;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trana, &tranb, &isgn, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &scale))
            ){
                LAPACKE_strsyl(matrix_layout, trana, tranb, isgn, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_ptr(scale));
            }
        break; }
        case dtrsyl: {
            int matrix_layout; char trana; char tranb; int isgn; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; c_binary scale;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trana, &tranb, &isgn, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &scale))
            ){
                LAPACKE_dtrsyl(matrix_layout, trana, tranb, isgn, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_ptr(scale));
            }
        break; }
        case ctrsyl: {
            int matrix_layout; char trana; char tranb; int isgn; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; c_binary scale;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trana, &tranb, &isgn, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &scale))
            ){
                LAPACKE_ctrsyl(matrix_layout, trana, tranb, isgn, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_ptr(scale));
            }
        break; }
        case ztrsyl: {
            int matrix_layout; char trana; char tranb; int isgn; int m; int n; cste_c_binary a; int lda; cste_c_binary b; int ldb; c_binary c; int ldc; c_binary scale;
            
            if( !(error = narg == 13? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &trana, &tranb, &isgn, &m, &n, &a, &lda, &b, &ldb, &c, &ldc, &scale))
            ){
                LAPACKE_ztrsyl(matrix_layout, trana, tranb, isgn, m, n, get_cste_ptr(a), lda, get_cste_ptr(b), ldb, get_ptr(c), ldc, get_ptr(scale));
            }
        break; }
        case strtri: {
            int matrix_layout; char uplo; char diag; int n; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &diag, &n, &a, &lda))
            ){
                LAPACKE_strtri(matrix_layout, uplo, diag, n, get_ptr(a), lda);
            }
        break; }
        case dtrtri: {
            int matrix_layout; char uplo; char diag; int n; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &diag, &n, &a, &lda))
            ){
                LAPACKE_dtrtri(matrix_layout, uplo, diag, n, get_ptr(a), lda);
            }
        break; }
        case ctrtri: {
            int matrix_layout; char uplo; char diag; int n; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &diag, &n, &a, &lda))
            ){
                LAPACKE_ctrtri(matrix_layout, uplo, diag, n, get_ptr(a), lda);
            }
        break; }
        case ztrtri: {
            int matrix_layout; char uplo; char diag; int n; c_binary a; int lda;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &diag, &n, &a, &lda))
            ){
                LAPACKE_ztrtri(matrix_layout, uplo, diag, n, get_ptr(a), lda);
            }
        break; }
        case strtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_strtrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case dtrtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_dtrtrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case ctrtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_ctrtrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case ztrtrs: {
            int matrix_layout; char uplo; char trans; char diag; int n; int nrhs; cste_c_binary a; int lda; c_binary b; int ldb;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &trans, &diag, &n, &nrhs, &a, &lda, &b, &ldb))
            ){
                LAPACKE_ztrtrs(matrix_layout, uplo, trans, diag, n, nrhs, get_cste_ptr(a), lda, get_ptr(b), ldb);
            }
        break; }
        case strttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary a; int lda; c_binary arf;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a, &lda, &arf))
            ){
                LAPACKE_strttf(matrix_layout, transr, uplo, n, get_cste_ptr(a), lda, get_ptr(arf));
            }
        break; }
        case dtrttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary a; int lda; c_binary arf;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a, &lda, &arf))
            ){
                LAPACKE_dtrttf(matrix_layout, transr, uplo, n, get_cste_ptr(a), lda, get_ptr(arf));
            }
        break; }
        case ctrttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary a; int lda; c_binary arf;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a, &lda, &arf))
            ){
                LAPACKE_ctrttf(matrix_layout, transr, uplo, n, get_cste_ptr(a), lda, get_ptr(arf));
            }
        break; }
        case ztrttf: {
            int matrix_layout; char transr; char uplo; int n; cste_c_binary a; int lda; c_binary arf;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &transr, &uplo, &n, &a, &lda, &arf))
            ){
                LAPACKE_ztrttf(matrix_layout, transr, uplo, n, get_cste_ptr(a), lda, get_ptr(arf));
            }
        break; }
        case strttp: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ap))
            ){
                LAPACKE_strttp(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(ap));
            }
        break; }
        case dtrttp: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ap))
            ){
                LAPACKE_dtrttp(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(ap));
            }
        break; }
        case ctrttp: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ap))
            ){
                LAPACKE_ctrttp(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(ap));
            }
        break; }
        case ztrttp: {
            int matrix_layout; char uplo; int n; cste_c_binary a; int lda; c_binary ap;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_int, e_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &ap))
            ){
                LAPACKE_ztrttp(matrix_layout, uplo, n, get_cste_ptr(a), lda, get_ptr(ap));
            }
        break; }
        case stzrzf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_stzrzf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case dtzrzf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_dtzrzf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case ctzrzf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_ctzrzf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case ztzrzf: {
            int matrix_layout; int m; int n; c_binary a; int lda; c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_ptr, e_int, e_ptr, e_end}, &matrix_layout, &m, &n, &a, &lda, &tau))
            ){
                LAPACKE_ztzrzf(matrix_layout, m, n, get_ptr(a), lda, get_ptr(tau));
            }
        break; }
        case cungbr: {
            int matrix_layout; char vect; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &vect, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cungbr(matrix_layout, vect, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case zungbr: {
            int matrix_layout; char vect; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 8? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &vect, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zungbr(matrix_layout, vect, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case cunghr: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_cunghr(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case zunghr: {
            int matrix_layout; int n; int ilo; int ihi; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &n, &ilo, &ihi, &a, &lda, &tau))
            ){
                LAPACKE_zunghr(matrix_layout, n, ilo, ihi, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case cunglq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cunglq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case zunglq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zunglq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case cungql: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cungql(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case zungql: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zungql(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case cungqr: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cungqr(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case zungqr: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zungqr(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case cungrq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_cungrq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case zungrq: {
            int matrix_layout; int m; int n; int k; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_int, e_int, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &m, &n, &k, &a, &lda, &tau))
            ){
                LAPACKE_zungrq(matrix_layout, m, n, k, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case cungtr: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &tau))
            ){
                LAPACKE_cungtr(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case zungtr: {
            int matrix_layout; char uplo; int n; c_binary a; int lda; cste_c_binary tau;
            
            if( !(error = narg == 6? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_ptr, e_int, e_cste_ptr, e_end}, &matrix_layout, &uplo, &n, &a, &lda, &tau))
            ){
                LAPACKE_zungtr(matrix_layout, uplo, n, get_ptr(a), lda, get_cste_ptr(tau));
            }
        break; }
        case cunmbr: {
            int matrix_layout; char vect; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmbr(matrix_layout, vect, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case zunmbr: {
            int matrix_layout; char vect; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &vect, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmbr(matrix_layout, vect, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case cunmhr: {
            int matrix_layout; char side; char trans; int m; int n; int ilo; int ihi; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &ilo, &ihi, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmhr(matrix_layout, side, trans, m, n, ilo, ihi, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case zunmhr: {
            int matrix_layout; char side; char trans; int m; int n; int ilo; int ihi; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &ilo, &ihi, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmhr(matrix_layout, side, trans, m, n, ilo, ihi, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case cunmlq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmlq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case zunmlq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmlq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case cunmql: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmql(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case zunmql: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmql(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case cunmqr: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmqr(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case zunmqr: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmqr(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case cunmrq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmrq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case zunmrq: {
            int matrix_layout; char side; char trans; int m; int n; int k; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmrq(matrix_layout, side, trans, m, n, k, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case cunmrz: {
            int matrix_layout; char side; char trans; int m; int n; int k; int l; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &l, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmrz(matrix_layout, side, trans, m, n, k, l, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case zunmrz: {
            int matrix_layout; char side; char trans; int m; int n; int k; int l; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 12? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_int, e_int, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &trans, &m, &n, &k, &l, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmrz(matrix_layout, side, trans, m, n, k, l, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case cunmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_cunmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case zunmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary a; int lda; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 11? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_int, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &a, &lda, &tau, &c, &ldc))
            ){
                LAPACKE_zunmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(a), lda, get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case cupgtr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary tau; c_binary q; int ldq;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &tau, &q, &ldq))
            ){
                LAPACKE_cupgtr(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(q), ldq);
            }
        break; }
        case zupgtr: {
            int matrix_layout; char uplo; int n; cste_c_binary ap; cste_c_binary tau; c_binary q; int ldq;
            
            if( !(error = narg == 7? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &uplo, &n, &ap, &tau, &q, &ldq))
            ){
                LAPACKE_zupgtr(matrix_layout, uplo, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(q), ldq);
            }
        break; }
        case cupmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary ap; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &ap, &tau, &c, &ldc))
            ){
                LAPACKE_cupmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }
        case zupmtr: {
            int matrix_layout; char side; char uplo; char trans; int m; int n; cste_c_binary ap; cste_c_binary tau; c_binary c; int ldc;
            
            if( !(error = narg == 10? 0:ERROR_N_ARG)
                && !(error = translate(env, elements, (etypes[]) {e_layout, e_char, e_char, e_char, e_int, e_int, e_cste_ptr, e_cste_ptr, e_ptr, e_int, e_end}, &matrix_layout, &side, &uplo, &trans, &m, &n, &ap, &tau, &c, &ldc))
            ){
                LAPACKE_zupmtr(matrix_layout, side, uplo, trans, m, n, get_cste_ptr(ap), get_cste_ptr(tau), get_ptr(c), ldc);
            }
        break; }

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
        case 1 ... 19:{
            char buff[50];
            sprintf(buff, "Could not translate argument %i.", error - 1);
            return enif_raise_exception(env, enif_make_atom(env, buff));
        break;}
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


ERL_NIF_TERM blas_hash_nif(ErlNifEnv* env, int argc, const ERL_NIF_TERM* argv){
    int max_len = 50;
    char name[max_len];

    if(!enif_get_atom(env, argv[0], name, max_len-1, ERL_NIF_LATIN1)){
        return enif_make_badarg(env);
    }

    unsigned long h = hash(name);

    return enif_make_uint64(env, h);
}

int load_blas(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info){
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
    {"hash", 1, blas_hash_nif},

    {"dirty_unwrapper", 2, unwrapper, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"clean_unwrapper", 2, unwrapper, 0}
};


ERL_NIF_INIT(blas, nif_funcs, load_blas, NULL, NULL, NULL)