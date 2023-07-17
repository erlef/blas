program MainProgram
  use PipeModule
  implicit none

  integer :: pipe, a, b
  a = 42
  b = 1

  print *, "At start ", a, b

  call pipe_open_write("mpype", pipe)
  call pipe_write_int(a, pipe)
  call pipe_write_int(b, pipe)
  call pipe_close(pipe)

  call pipe_open_read("mpype", pipe)
  call pipe_read_int(a, pipe)
  call pipe_read_int(b, pipe)
  call pipe_close(pipe)

  print *, "At end   ", a, b
end program MainProgram