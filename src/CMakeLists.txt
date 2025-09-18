cmake_minimum_required(VERSION 3.10)

project(rknn_ppocrv5_lib) # Đổi tên project cho rõ ràng

# --- CÁC CẤU HÌNH BUILD GIỮ NGUYÊN ---
if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()

find_package(OpenCV REQUIRED)

# Thêm cờ -fPIC để tạo Position-Independent Code, bắt buộc cho thư viện động
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

# --- CÁC THƯ VIỆN PHỤ THUỘC GIỮ NGUYÊN ---
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/ 3rdparty.out)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../utils/ utils.out)


# =================== THAY ĐỔI CHÍNH BẮT ĐẦU TỪ ĐÂY ===================

# 1. Định nghĩa tên thư viện
set(API_LIBRARY_NAME "ppocrv5_api")

# 2. Liệt kê các file nguồn cho thư viện (TẤT CẢ TRỪ main.cc)
set(API_SOURCES
    postprocess.cc
    clipper.cc
    rknpu2/ppocrv5.cc
)

# 3. Tạo thư viện động (.so) từ các file nguồn
add_library(${API_LIBRARY_NAME} SHARED ${API_SOURCES})

# 4. Chỉ định các thư mục include và link với các thư viện phụ thuộc
target_link_libraries(${API_LIBRARY_NAME}
    PUBLIC # Dùng PUBLIC để các thuộc tính này được truyền cho target link với nó
        imageutils
        fileutils
        ${OpenCV_LIBS}
        # ${LIBRGA}
        ${LIBRKNNRT}
)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Threads REQUIRED)
    target_link_libraries(${API_LIBRARY_NAME} PUBLIC Threads::Threads)
endif()

target_include_directories(${API_LIBRARY_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/rknpu2 # Thêm thư mục con
    ${LIBRKNNRT_INCLUDES}
    ${LIBTIMER_INCLUDES}
    # ${RGA_INCLUDES}
)

# 5. Tạo một file thực thi VÍ DỤ từ main.cc để kiểm tra thư viện
add_executable(ppocrv5_example main.cc)
target_link_libraries(ppocrv5_example PRIVATE ${API_LIBRARY_NAME})


# 6. Định nghĩa quy trình cài đặt (make install)
include(GNUInstallDirs)

# Cài đặt file thư viện .so vào thư mục lib
install(TARGETS ${API_LIBRARY_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Cài đặt file header .h công khai vào thư mục include
install(FILES 
    ppocrv5.h
    dict_ppocrv5.h
    en_dict_ppocrv5.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Cài đặt file thực thi ví dụ vào thư mục bin
install(TARGETS ppocrv5_example
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Cài đặt các model đi kèm
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../model/general_ocr_002.png DESTINATION model)
set(file_path ${CMAKE_CURRENT_SOURCE_DIR}/../model/PP-OCRv5_mobile_det.rknn)
if (EXISTS ${file_path})
    install(FILES ${file_path} DESTINATION model)
endif()

set(file_path ${CMAKE_CURRENT_SOURCE_DIR}/../model/PP-OCRv5_mobile_rec.rknn)
if (EXISTS ${file_path})
    install(FILES ${file_path} DESTINATION model)
endif()