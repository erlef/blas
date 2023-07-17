-module(named_pipe_reader).
-export([start/1]).

start(PipeName) ->
    {ok, Binary} = file:read_file(PipeName),
    Content = process(Binary),
    Result  = modify(Content),
    ok      = file:write_file(PipeName, encode(Result)).
    

modify(List)->
    lists:map(fun(X)-> X + 2 end, List).

encode(List)->
    encode(List, <<>>).

encode([], Binary) -> Binary;
encode([H|T], Binary) ->
    Encoded = 
    case H of
        I when is_integer(I) -> <<32:32/native-integer, 0:32/native-integer, H:32/native-integer>>
    end,
    encode(T, <<Binary/binary, Encoded/binary>>).


process(Data)->
    process(Data, []).


process(<<>>, Processed)->
    lists:reverse(Processed);

process(Data, Processed)->
    << _:32/native-integer, Type:32/native-integer, Rest/binary>> = Data,
    case Type of 
        0 -> % Integer.
            <<Integer:32/native-integer, Next/binary>> = Rest,
            process(Next, [Integer] ++ Processed)
    end.