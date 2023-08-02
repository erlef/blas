

typedef enum sizes {s_bytes=4, d_bytes=8, c_bytes=8, z_bytes=16, no_bytes=0} size_in_bytes;

typedef enum BLAS_NAMES {
   sdsdot = 6954012548918,
   dsdot = 210710387267,
   sdot = 6385686335,
   ddot = 6385147280,
   cdotu = 210708674436,
   cdotc = 210708674418,
   zdotu = 210735950619,
   zdotc = 210735950601,
   snrm2 = 210728011511,
   sasum = 210727545742,
   dnrm2 = 210710222696,
   dasum = 210709756927,
   scnrm2 = 6954011198426,
   scasum = 6954010732657,
   dznrm2 = 6953451443714,
   dzasum = 6953450977945,
   isamax = 6953638346280,
   idamax = 6953620557465,
   icamax = 6953619371544,
   izamax = 6953646647727,
   sswap = 210728196307,
   scopy = 210727613107,
   saxpy = 210727551034,
   dswap = 210710407492,
   dcopy = 210709824292,
   daxpy = 210709762219,
   cswap = 210709221571,
   ccopy = 210708638371,
   caxpy = 210708576298,
   zswap = 210736497754,
   zcopy = 210735914554,
   zaxpy = 210735852481,
   srotg = 210728152276,
   srotmg = 6954029025409,
   srot = 6385701581,
   srotm = 210728152282,
   drotg = 210710363461,
   drotmg = 6953441994514,
   drot = 6385162526,
   drotm = 210710363467,
   sscal = 210728174523,
   dscal = 210710385708,
   cscal = 210709199787,
   zscal = 210736475970,
   csscal = 6953404169886,
   zdscal = 6954286495110,
   sgemv = 210727745863,
   sgbmv = 210727742596,
   strmv = 210728227201,
   stbmv = 210728209777,
   stpmv = 210728225023,
   strsv = 210728227399,
   stbsv = 210728209975,
   stpsv = 210728225221,
   dgemv = 210709957048,
   dgbmv = 210709953781,
   dtrmv = 210710438386,
   dtbmv = 210710420962,
   dtpmv = 210710436208,
   dtrsv = 210710438584,
   dtbsv = 210710421160,
   dtpsv = 210710436406,
   cgemv = 210708771127,
   cgbmv = 210708767860,
   ctrmv = 210709252465,
   ctbmv = 210709235041,
   ctpmv = 210709250287,
   ctrsv = 210709252663,
   ctbsv = 210709235239,
   ctpsv = 210709250485,
   zgemv = 210736047310,
   zgbmv = 210736044043,
   ztrmv = 210736528648,
   ztbmv = 210736511224,
   ztpmv = 210736526470,
   ztrsv = 210736528846,
   ztbsv = 210736511422,
   ztpsv = 210736526668,
   ssymv = 210728198887,
   ssbmv = 210728173840,
   sspmv = 210728189086,
   sger = 6385689270,
   ssyr = 6385702998,
   sspr = 6385702701,
   ssyr2 = 210728198984,
   sspr2 = 210728189183,
   dsymv = 210710410072,
   dsbmv = 210710385025,
   dspmv = 210710400271,
   dger = 6385150215,
   dsyr = 6385163943,
   dspr = 6385163646,
   dsyr2 = 210710410169,
   dspr2 = 210710400368,
   chemv = 210708807064,
   chbmv = 210708803797,
   chpmv = 210708819043,
   cgeru = 210708771291,
   cgerc = 210708771273,
   cher = 6385115367,
   chpr = 6385115730,
   cher2 = 210708807161,
   chpr2 = 210708819140,
   zhemv = 210736083247,
   zhbmv = 210736079980,
   zhpmv = 210736095226,
   zgeru = 210736047474,
   zgerc = 210736047456,
   zher = 6385941918,
   zhpr = 6385942281,
   zher2 = 210736083344,
   zhpr2 = 210736095323,
   sgemm = 210727745854,
   ssymm = 210728198878,
   ssyrk = 210728199041,
   ssyr2k = 6954030566579,
   strmm = 210728227192,
   strsm = 210728227390,
   dgemm = 210709957039,
   dsymm = 210710410063,
   dsyrk = 210710410226,
   dsyr2k = 6953443535684,
   dtrmm = 210710438377,
   dtrsm = 210710438575,
   cgemm = 210708771118,
   csymm = 210709224142,
   csyrk = 210709224305,
   csyr2k = 6953404400291,
   ctrmm = 210709252456,
   ctrsm = 210709252654,
   zgemm = 210736047301,
   zsymm = 210736500325,
   zsyrk = 210736500488,
   zsyr2k = 6954304514330,
   ztrmm = 210736528639,
   ztrsm = 210736528837,
   chemm = 210708807055,
   cherk = 210708807218,
   cher2k = 6953390636420,
   zhemm = 210736083238,
   zherk = 210736083401,
   zher2k = 6954290750459,
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