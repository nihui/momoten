# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

foreach(test_case "20261020;26;10;20" "20260809;26;8;9" "20000229;0;2;29" "20991231;99;12;31")
    list(GET test_case 0 MOMOTEN_RELEASE_VERSION)
    unset(MOMOTEN_RELEASE_VERSION CACHE)
    include("${SOURCE_ROOT}/cmake/momoten_version.cmake")
    list(GET test_case 1 expected_major)
    list(GET test_case 2 expected_minor)
    list(GET test_case 3 expected_patch)
    if(NOT MOMOTEN_DRIVER_VERSION_MAJOR EQUAL expected_major OR
       NOT MOMOTEN_DRIVER_VERSION_MINOR EQUAL expected_minor OR
       NOT MOMOTEN_DRIVER_VERSION_PATCH EQUAL expected_patch)
        message(FATAL_ERROR "Incorrect driver version for ${MOMOTEN_RELEASE_VERSION}")
    endif()
endforeach()

foreach(invalid "" "2026102" "202610200" "v20261020" "2026.10.20" "19991231" "21000101"
                "20260020" "20261320" "20261000" "20261032" "20260431" "20260229" "20010229")
    execute_process(COMMAND "${CMAKE_COMMAND}" "-DMOMOTEN_RELEASE_VERSION=${invalid}"
        -P "${SOURCE_ROOT}/cmake/momoten_version.cmake"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(result EQUAL 0 OR NOT error MATCHES "MOMOTEN_RELEASE_VERSION must be a valid YYYYMMDD date")
        message(FATAL_ERROR "Invalid release version was not rejected: '${invalid}'\n${output}${error}")
    endif()
endforeach()

# SOURCE_DATE_EPOCH makes the default-date test independent of the wall clock.
unset(MOMOTEN_RELEASE_VERSION CACHE)
unset(MOMOTEN_RELEASE_VERSION)
set(ENV{SOURCE_DATE_EPOCH} 1792454400)
string(TIMESTAMP expected_date "%Y%m%d" UTC)
include("${SOURCE_ROOT}/cmake/momoten_version.cmake")
if(NOT MOMOTEN_RELEASE_VERSION STREQUAL expected_date)
    message(FATAL_ERROR "The default release version is not the UTC configure date")
endif()
set(ENV{SOURCE_DATE_EPOCH} 1792540800)
include("${SOURCE_ROOT}/cmake/momoten_version.cmake")
if(NOT MOMOTEN_RELEASE_VERSION STREQUAL expected_date)
    message(FATAL_ERROR "Reconfiguration changed the cached release version")
endif()
