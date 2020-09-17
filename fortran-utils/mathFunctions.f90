module mathUtilities
  use, intrinsic :: iso_c_binding
contains

  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ! DESCRIPTION
  !> @brief
  !> Compute the Planck Taper window function
  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  elemental function planckTaper(x) return(r)
    real(C_DOUBLE), intent(in) :: x
    real(C_DOUBLE) :: r
    
    
  end function planckTaper
    
end module mathUtilities
