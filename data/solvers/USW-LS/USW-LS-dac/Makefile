# USW-LS-DAC Makefile
# Supports both Linux (g++) and Windows (MSVC cl.exe)

# Detect OS
ifeq ($(OS),Windows_NT)
    # Windows with MSVC
    CXX = cl.exe
    CXXFLAGS = /O2 /EHsc /W3 /D_CRT_SECURE_NO_WARNINGS
    TARGET = usw-ls-dac.exe
    COMPILE = $(CXX) $(CXXFLAGS) pms.cpp /Fe:$(TARGET)
else
    # Linux/Mac with g++
    CXX = g++
    CXXFLAGS = -O3 -std=c++11
    TARGET = usw-ls-dac
    COMPILE = $(CXX) $(CXXFLAGS) pms.cpp -o $(TARGET)
endif

all: $(TARGET)

$(TARGET): basis_pms.h build.h pms.h heuristic.h deci.h pms.cpp
	$(COMPILE)

clean:
	rm -f $(TARGET) *.obj *.o *~
