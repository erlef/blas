-module(timeEst).
-export([benchmark/1, benchmark/0, n_elements/1]).


benchmark(Size) ->
    GenMatrix = fun(N) -> 
        blas:new(float64, lists:seq(1,N*N))
    end,
    GenTuple = fun(N) ->
        {dgemm, blasColMajor, blasNoTrans, blasNoTrans, N,N,N, 1.0, GenMatrix(N), N, GenMatrix(N), N, 1.0, GenMatrix(N), N}
    end,

    Tuple = GenTuple(Size),
    {T, ok} = timer:tc(blas, run, [Tuple, dirty]),
    T.

normsum(L)->
    % Do a weighed normalised sum of the elements of a list, with weigths: 1, ... , 0.1
    N_elems = length(L),
    if 
        N_elems == 1 ->
            lists:nth(1,L);
        true ->
            Weigths = [I/N_elems || I <- lists:seq(N_elems, 1, -1)],
            Sum     = lists:foldl(
                fun({E,W}, S) ->
                    S + E*W 
                end,
                0,
                lists:zip(L, Weigths)
            ),
            Result = round(Sum / lists:foldl(fun(V,A)-> V+A end, 0, Weigths)),
            Result
    end.

benchmark()->
    % Reach 1ms.
    Iterate = fun(N)->
        T = benchmark(N),
        %io:format("Size ~w, Time ~w ~n", [N, T]),
        round(max(min(10, 1000.0/T), 0.1) * N) % Do not explode/collapse size: limit ratio in range(0.1, 10).
    end,

    R = normsum(lists:foldl(fun (_, L) -> [Iterate(normsum(L))|L] end, [Iterate(20)], lists:seq(1, 20))),
	%io:format("~w ~n", [R]),
	math:log(R*R*R).

n_elements(BlasOp)->
	V = case BlasOp of
		{caxpy,N,_,_,_,_,_} -> N;
		{ccopy,N,_,_,_,_} -> N;
		{cdotc,N,_,_,_,_,_} -> N;
		{cdotu,N,_,_,_,_,_} -> N;
		{cgbmv,_,_,M,N,_,_,_,_,_,_,_,_,_,_} -> M*N;
		{cgemm,_,_,_,M,N,K,_,_,_,_,_,_,_,_} -> M*N*K;
		{cgemv,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{cgerc,_,M,N,_,_,_,_,_,_,_} -> M*N;
		{cgeru,_,M,N,_,_,_,_,_,_,_} -> M*N;
		{chbmv,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{chemm,_,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{chemv,_,_,N,_,_,_,_,_,_,_,_} -> N;
		{cher,_,_,N,_,_,_,_,_} -> N;
		{cher2,_,_,N,_,_,_,_,_,_,_} -> N;
		{cher2k,_,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{cherk,_,_,_,N,K,_,_,_,_,_,_} -> N*K;
		{chpmv,_,_,N,_,_,_,_,_,_,_} -> N;
		{chpr,_,_,N,_,_,_,_} -> N;
		{chpr2,_,_,N,_,_,_,_,_,_} -> N;
		{cscal,N,_,_,_} -> N;
		{csscal,N,_,_,_} -> N;
		{cswap,N,_,_,_,_} -> N;
		{csymm,_,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{csyr2k,_,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{csyrk,_,_,_,N,K,_,_,_,_,_,_} -> N*K;
		{ctbmv,_,_,_,_,N,K,_,_,_,_} -> N*K;
		{ctbsv,_,_,_,_,N,K,_,_,_,_} -> N*K;
		{ctpmv,_,_,_,_,N,_,_,_} -> N;
		{ctpsv,_,_,_,_,N,_,_,_} -> N;
		{ctrmm,_,_,_,_,_,M,N,_,_,_,_,_} -> M*N;
		{ctrmv,_,_,_,_,N,_,_,_,_} -> N;
		{ctrsm,_,_,_,_,_,M,N,_,_,_,_,_} -> M*N;
		{ctrsv,_,_,_,_,N,_,_,_,_} -> N;
		{dasum,N,_,_} -> N;
		{daxpy,N,_,_,_,_,_} -> N;
		{dcopy,N,_,_,_,_} -> N;
		{ddot,N,_,_,_,_} -> N;
		{dgbmv,_,_,M,N,_,_,_,_,_,_,_,_,_,_} -> M*N;
		{dgemm,_,_,_,M,N,K,_,_,_,_,_,_,_,_} -> M*N*K;
		{dgemv,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{dger,_,M,N,_,_,_,_,_,_,_} -> M*N;
		{dnrm2,N,_,_} -> N;
		{drot,N,_,_,_,_,_,_} -> N;
		{drotg,_,_,_,_} -> 0;
		{drotm,N,_,_,_,_,_} -> N;
		{drotmg,_,_,_,_,_} -> 0;
		{dsbmv,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{dscal,N,_,_,_} -> N;
		{dsdot,N,_,_,_,_} -> N;
		{dspmv,_,_,N,_,_,_,_,_,_,_} -> N;
		{dspr,_,_,N,_,_,_,_} -> N;
		{dspr2,_,_,N,_,_,_,_,_,_} -> N;
		{dswap,N,_,_,_,_} -> N;
		{dsymm,_,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{dsymv,_,_,N,_,_,_,_,_,_,_,_} -> N;
		{dsyr,_,_,N,_,_,_,_,_} -> N;
		{dsyr2,_,_,N,_,_,_,_,_,_,_} -> N;
		{dsyr2k,_,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{dsyrk,_,_,_,N,K,_,_,_,_,_,_} -> N*K;
		{dtbmv,_,_,_,_,N,K,_,_,_,_} -> N*K;
		{dtbsv,_,_,_,_,N,K,_,_,_,_} -> N*K;
		{dtpmv,_,_,_,_,N,_,_,_} -> N;
		{dtpsv,_,_,_,_,N,_,_,_} -> N;
		{dtrmm,_,_,_,_,_,M,N,_,_,_,_,_} -> M*N;
		{dtrmv,_,_,_,_,N,_,_,_,_} -> N;
		{dtrsm,_,_,_,_,_,M,N,_,_,_,_,_} -> M*N;
		{dtrsv,_,_,_,_,N,_,_,_,_} -> N;
		{dzasum,N,_,_} -> N;
		{dznrm2,N,_,_} -> N;
		{icamax,N,_,_} -> N;
		{idamax,N,_,_} -> N;
		{isamax,N,_,_} -> N;
		{izamax,N,_,_} -> N;
		{sasum,N,_,_} -> N;
		{saxpy,N,_,_,_,_,_} -> N;
		{scasum,N,_,_} -> N;
		{scnrm2,N,_,_} -> N;
		{scopy,N,_,_,_,_} -> N;
		{sdot,N,_,_,_,_} -> N;
		{sdsdot,N,_,_,_,_,_} -> N;
		{sgbmv,_,_,M,N,_,_,_,_,_,_,_,_,_,_} -> M*N;
		{sgemm,_,_,_,M,N,K,_,_,_,_,_,_,_,_} -> M*N*K;
		{sgemv,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{sger,_,M,N,_,_,_,_,_,_,_} -> M*N;
		{snrm2,N,_,_} -> N;
		{srot,N,_,_,_,_,_,_} -> N;
		{srotg,_,_,_,_} -> 0;
		{srotm,N,_,_,_,_,_} -> N;
		{srotmg,_,_,_,_,_} -> 0;
		{ssbmv,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{sscal,N,_,_,_} -> N;
		{sspmv,_,_,N,_,_,_,_,_,_,_} -> N;
		{sspr,_,_,N,_,_,_,_} -> N;
		{sspr2,_,_,N,_,_,_,_,_,_} -> N;
		{sswap,N,_,_,_,_} -> N;
		{ssymm,_,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{ssymv,_,_,N,_,_,_,_,_,_,_,_} -> N;
		{ssyr,_,_,N,_,_,_,_,_} -> N;
		{ssyr2,_,_,N,_,_,_,_,_,_,_} -> N;
		{ssyr2k,_,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{ssyrk,_,_,_,N,K,_,_,_,_,_,_} -> N*K;
		{stbmv,_,_,_,_,N,K,_,_,_,_} -> N*K;
		{stbsv,_,_,_,_,N,K,_,_,_,_} -> N*K;
		{stpmv,_,_,_,_,N,_,_,_} -> N;
		{stpsv,_,_,_,_,N,_,_,_} -> N;
		{strmm,_,_,_,_,_,M,N,_,_,_,_,_} -> M*N;
		{strmv,_,_,_,_,N,_,_,_,_} -> N;
		{strsm,_,_,_,_,_,M,N,_,_,_,_,_} -> M*N;
		{strsv,_,_,_,_,N,_,_,_,_} -> N;
		{zaxpy,N,_,_,_,_,_} -> N;
		{zcopy,N,_,_,_,_} -> N;
		{zdotc,N,_,_,_,_,_} -> N;
		{zdotu,N,_,_,_,_,_} -> N;
		{zdscal,N,_,_,_} -> N;
		{zgbmv,_,_,M,N,_,_,_,_,_,_,_,_,_,_} -> M*N;
		{zgemm,_,_,_,M,N,K,_,_,_,_,_,_,_,_} -> M*N*K;
		{zgemv,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{zgerc,_,M,N,_,_,_,_,_,_,_} -> M*N;
		{zgeru,_,M,N,_,_,_,_,_,_,_} -> M*N;
		{zhbmv,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{zhemm,_,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{zhemv,_,_,N,_,_,_,_,_,_,_,_} -> N;
		{zher,_,_,N,_,_,_,_,_} -> N;
		{zher2,_,_,N,_,_,_,_,_,_,_} -> N;
		{zher2k,_,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{zherk,_,_,_,N,K,_,_,_,_,_,_} -> N*K;
		{zhpmv,_,_,N,_,_,_,_,_,_,_} -> N;
		{zhpr,_,_,N,_,_,_,_} -> N;
		{zhpr2,_,_,N,_,_,_,_,_,_} -> N;
		{zscal,N,_,_,_} -> N;
		{zswap,N,_,_,_,_} -> N;
		{zsymm,_,_,_,M,N,_,_,_,_,_,_,_,_} -> M*N;
		{zsyr2k,_,_,_,N,K,_,_,_,_,_,_,_,_} -> N*K;
		{zsyrk,_,_,_,N,K,_,_,_,_,_,_} -> N*K;
		{ztbmv,_,_,_,_,N,K,_,_,_,_} -> N*K;
		{ztbsv,_,_,_,_,N,K,_,_,_,_} -> N*K;
		{ztpmv,_,_,_,_,N,_,_,_} -> N;
		{ztpsv,_,_,_,_,N,_,_,_} -> N;
		{ztrmm,_,_,_,_,_,M,N,_,_,_,_,_} -> M*N;
		{ztrmv,_,_,_,_,N,_,_,_,_} -> N;
		{ztrsm,_,_,_,_,_,M,N,_,_,_,_,_} -> M*N;
		{ztrsv,_,_,_,_,N,_,_,_,_} -> N;
		_ -> -1
	end,
	if V > 0 -> math:log(V+1); true -> V end.