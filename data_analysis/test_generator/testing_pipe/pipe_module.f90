module PipeModule

! Each message is of type: size(32 bits), type(32bit int), binary of size bits)

  implicit none
  public

  character(len=:), allocatable :: pipe_name

contains

  subroutine pipe_open_write(pipe_name, pipe)
    implicit none
    character(len=*), intent(in) :: pipe_name
    integer, intent(out) :: pipe
    integer :: ierr

    ! Open the named pipe for writing
    open(newunit=pipe, file=pipe_name, status='replace', access='stream', action='write', iostat=ierr)
    if (ierr /= 0) then
      write(*, *) "Error opening the named pipe."
      return
    end if

  end subroutine 
  
  subroutine pipe_open_read(pipe_name, pipe)
    implicit none
    character(len=*), intent(in) :: pipe_name
    integer, intent(out) :: pipe
    integer :: ierr

    ! Open the named pipe for writing
    open(newunit=pipe, file=pipe_name, status='replace', access='stream', action='read', iostat=ierr)
    if (ierr /= 0) then
      write(*, *) "Error opening the named pipe."
      return
    end if

  end subroutine


  subroutine pipe_close(pipe)
    implicit none
    integer, intent(in) :: pipe
    close(pipe)
  end subroutine 


  subroutine pipe_write_int(param, pipe)
    implicit none
    integer, intent(in) :: param, pipe

    ! Write the parameters to the pipe
    write(pipe) 32
    write(pipe) 0
    write(pipe) param

  end subroutine

  subroutine pipe_read_int(param, pipe)
    implicit none
    integer, intent(out) :: param
    integer, intent(in)  :: pipe
    integer :: discard

    ! Write the parameters to the pipe
    read(pipe) discard
    read(pipe) discard
    read(pipe) param

  end subroutine 

end module PipeModule