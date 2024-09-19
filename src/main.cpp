#include <iostream>
#include <blt/math/matrix.h>
#include <blt/math/log_util.h>
#include <valarray>
#include "blt/std/assert.h"

void test_math()
{
    blt::generalized_matrix<float, 1, 4> input{1, -1, -1, 1};
    blt::generalized_matrix<float, 1, 3> output{1, 1, 1};
    blt::generalized_matrix<float, 4, 3> expected{
            blt::vec4{1, -1, -1, 1},
            blt::vec4{1, -1, -1, 1},
            blt::vec4{1, -1, -1, 1}
    };
    
    auto w_matrix = input.transpose() * output;
    BLT_ASSERT(w_matrix == expected && "MATH FAILURE");
}

constexpr blt::u32 input_count = 5;
constexpr blt::u32 output_count = 4;

using input_t = blt::generalized_matrix<float, 1, input_count>;
using output_t = blt::generalized_matrix<float, 1, output_count>;

input_t input_1{-1, 1, 1, 1, -1};
output_t output_1{1, 1, -1, 1};

int main()
{
    test_math();
    
    
    
    std::cout << output_1 << std::endl;
    std::cout << input_1.transpose() << std::endl;
    std::cout << input_1.transpose() * output_1 << std::endl;
    std::cout << "Hello World!" << std::endl;
}
