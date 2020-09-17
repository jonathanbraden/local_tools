! For this to be useful, I need to define a whole subset of special cases then choose the appropriate one based on the type of object passed in
! Is there a more elegant way to do this?

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!>@author
!> Jonathan Braden
!> University College London
!>
!> Reshape Fortran arrays without copying data via the use of pointers
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module reshapeArrays
  use constants
  use iso_c_binding, only : C_LOC, C_F_POINTER
  implicit none

contains

  ! DESCRIPTION
  !> @brief
  !> Takes as input an array of dimension > 1 and flattens it into a 1D array
  !> For this to work, the original multidimensional array must be contiguous
  subroutine flatten2D(in, out)
    real, dimension(:,:), pointer, contiguous :: in
    real, dimension(:), pointer :: out
    integer :: n

    n = size(in)
    out => in
  end subroutine flatten

  ! DESCRIPTION
  !> @brief
  !> Reshape a one-dimensional array into a multidimensional array without copying
  !>
  !> This function takes an array of doubles stored in a Fortran array (of arbitrary shape) and reshapes it
  !> by returning a pointer to the array with the given shape.
  !> Unlike the intrinsic reshape function, this does not required copying the array
  !>
  !> @param[in] in - The array of doubles to reshape
  !> @param[in] shape_ - the desired shape of the output array
  !>
  !> @returns aPtr - A pointer to the reshaped array
  function resizeArray_dble(in, shape_) result(aPtr)
    double precision, target :: in(1) !< Treat as assumed shape so pointer doesn't point to array descriptor
    integer, intent(in) :: shape_(:)
    double precision, pointer :: aPtr

    call C_F_POINTER(C_LOC(in), aPtr, shape_)
  end function resizeArray_dble
  
  function resizeArray_sgle(in, shape_) result(aPtr)
    real, target :: in(1)
    integer, in :: shape_(:)
    real, pointer :: aPtr

    call C_F_POINTER(C_LOC(in), aPtr, shape_)
  end function resizeArray_sgle
    
  function resizeArray_int
    integer, target :: in(1)
    integer, intent(in) :: shape_(:)
    real, pointer :: aPtr
    
    call C_F_POINTER(C_LOC(in), aPtr, shape_)
  end function resizeArray_int
  
end module reshapeArrays
