# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

# Keep the local default stable across reconfiguration of the same build tree.
if(NOT DEFINED MOMOTEN_RELEASE_VERSION)
    string(TIMESTAMP MOMOTEN_RELEASE_VERSION "%Y%m%d" UTC)
endif()
set(MOMOTEN_RELEASE_VERSION "${MOMOTEN_RELEASE_VERSION}" CACHE STRING
    "Release date in YYYYMMDD format (2000-2099)")

if(NOT MOMOTEN_RELEASE_VERSION MATCHES "^20[0-9][0-9][0-9][0-9][0-9][0-9]$")
    message(FATAL_ERROR "MOMOTEN_RELEASE_VERSION must be a valid YYYYMMDD date in 2000-2099")
endif()

# Vulkan's packed version fields cannot hold YYYYMMDD in the patch component.
# Encode the date as (year - 2000).month.day instead, e.g. 20261020 -> 26.10.20.
math(EXPR MOMOTEN_DRIVER_VERSION_MAJOR "${MOMOTEN_RELEASE_VERSION} / 10000 - 2000")
math(EXPR MOMOTEN_DRIVER_VERSION_MINOR "${MOMOTEN_RELEASE_VERSION} / 100 % 100")
math(EXPR MOMOTEN_DRIVER_VERSION_PATCH "${MOMOTEN_RELEASE_VERSION} % 100")
set(momoten_days_in_month 31)
if(MOMOTEN_DRIVER_VERSION_MINOR EQUAL 2)
    math(EXPR momoten_leap_year "${MOMOTEN_DRIVER_VERSION_MAJOR} % 4")
    if(momoten_leap_year EQUAL 0)
        set(momoten_days_in_month 29)
    else()
        set(momoten_days_in_month 28)
    endif()
elseif(MOMOTEN_DRIVER_VERSION_MINOR EQUAL 4 OR
       MOMOTEN_DRIVER_VERSION_MINOR EQUAL 6 OR
       MOMOTEN_DRIVER_VERSION_MINOR EQUAL 9 OR
       MOMOTEN_DRIVER_VERSION_MINOR EQUAL 11)
    set(momoten_days_in_month 30)
endif()
if(MOMOTEN_DRIVER_VERSION_MINOR LESS 1 OR MOMOTEN_DRIVER_VERSION_MINOR GREATER 12 OR
   MOMOTEN_DRIVER_VERSION_PATCH LESS 1 OR MOMOTEN_DRIVER_VERSION_PATCH GREATER momoten_days_in_month)
    message(FATAL_ERROR "MOMOTEN_RELEASE_VERSION must be a valid YYYYMMDD date in 2000-2099")
endif()
