-module(chain).

-export([ltb/2, btl/2]).

-define(IS_PAIR_LENGTH(List), (length(List) band 1) == 0 ).
-define(IS_SIZE_ALIGNED(Binary, ElementSize), size(Binary) rem ElementSize == 0).
-define(IS_PAIR_ELEMENTS(Binary, ElementSize), (round(size(Binary)/ElementSize) band 1) == 0).

ltb(Type, List)->
    case Type of
        int32 -> << <<V:32/native-integer>> || V <- List >>;
        int64 -> << <<V:32/native-integer>> || V <- List >>;
        S when S==s orelse S==float32 -> << <<V:32/native-float>> || V <- List >>;
        D when D==d orelse D==float64 -> << <<V:64/native-float>> || V <- List >>;
        C when (C==c orelse C==complex64)  andalso ?IS_PAIR_LENGTH(List) -> ltb(s, List); % Two elements per items
        Z when (Z==z orelse Z==complex128) andalso ?IS_PAIR_LENGTH(List) -> ltb(d, List)  % Two elements per items
    end.

btl(Type, Binary)->
    case Type of 
        int32 when ?IS_SIZE_ALIGNED(Binary, 4) -> [ V || <<V:32/native-integer>> <= Binary ];
        int64 when ?IS_SIZE_ALIGNED(Binary, 4) -> [ V || <<V:32/native-integer>> <= Binary ];
        S when (S==s orelse S==float32) andalso ?IS_SIZE_ALIGNED(Binary, 4) -> [ V || <<V:32/native-float>> <= Binary ];
        D when (D==d orelse D==float64) andalso ?IS_SIZE_ALIGNED(Binary, 8) -> [ V || <<V:64/native-float>> <= Binary ];
        C when (C==c orelse C==complex64)  andalso ?IS_PAIR_ELEMENTS(Binary, 4) -> btl(s, Binary);
        Z when (Z==z orelse Z==complex128) andalso ?IS_PAIR_ELEMENTS(Binary, 8) -> btl(d, Binary)
    end.