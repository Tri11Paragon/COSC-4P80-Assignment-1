#include <iostream>
#include <blt/math/matrix.h>
#include <blt/math/log_util.h>
#include "blt/std/assert.h"
#include <blt/format/boxing.h>

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

constexpr blt::u32 num_values = 4;
constexpr blt::u32 input_count = 5;
constexpr blt::u32 output_count = 4;

using input_t = blt::generalized_matrix<float, 1, input_count>;
using output_t = blt::generalized_matrix<float, 1, output_count>;
using weight_t = decltype(std::declval<input_t>().transpose() * std::declval<output_t>());
using crosstalk_t = blt::generalized_matrix<float, output_count, num_values>;

float crosstalk(const input_t& i, const input_t& j)
{
    return i * j.transpose();
}

input_t input_1{-1, 1, 1, 1, -1};
input_t input_2{-1, -1, -1, -1, 1};
input_t input_3{-1, -1, -1, 1, 1};
input_t input_4{1, 1, 1, 1, 1};

output_t output_1{1, 1, -1, 1};
output_t output_2{1, -1, -1, -1};
output_t output_3{-1, -1, 1, 1};
output_t output_4{-1, 1, 1, -1};

weight_t weight_1 = input_1.transpose() * output_1;
weight_t weight_2 = input_2.transpose() * output_2;
weight_t weight_3 = input_3.transpose() * output_3;
weight_t weight_4 = input_4.transpose() * output_4;

auto inputs = std::array{input_1, input_2, input_3, input_4};
auto outputs = std::array{output_1, output_2, output_3, output_4};

auto weight_total_a = weight_1 + weight_2 + weight_3;
auto weight_total_c = weight_total_a + weight_4;

crosstalk_t crosstalk_values{};

template<typename T, blt::u32 rows, blt::u32 columns>
blt::generalized_matrix<T, rows, columns> normalize(const blt::generalized_matrix<T, rows, columns>& in)
{
    blt::generalized_matrix<T, rows, columns> result;
    for (blt::u32 i = 0; i < columns; i++)
    {
        for (blt::u32 j = 0; j < rows; j++)
            result[i][j] = in[i][j] >= 0 ? 1 : -1;
    }
    return result;
}

auto calculate_recall()
{

}

void test_recall(blt::size_t index, const weight_t& associated_weights)
{
    auto& input = inputs[index];
    auto& output = outputs[index];
    
    auto output_recall = normalize(input * associated_weights);
    auto input_recall = normalize(output * associated_weights.transpose());
    
    if (output_recall != output)
    {
        BLT_ERROR_STREAM << "Output '" << index + 1 << "' recalled failed!" << '\n';
        BLT_WARN_STREAM << "\t- Found: " << output_recall.vec_from_column_row() << '\n';
        BLT_WARN_STREAM << "\t- Expected: " << output.vec_from_column_row() << '\n';
    } else
        BLT_INFO("Output '%ld' recall passed!", index + 1);
    
    if (input_recall != input)
    {
        BLT_ERROR_STREAM << "Input '" << index + 1 << "' recalled failed!" << "\n";
        BLT_WARN_STREAM << "\t- Found: " << input_recall.vec_from_column_row() << '\n';
        BLT_WARN_STREAM << "\t- Expected: " << input.vec_from_column_row() << '\n';
    } else
        BLT_INFO("Input '%ld' recall passed!", index + 1);
}

void part_a()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part A", 8);
    test_recall(0, weight_total_a);
    test_recall(1, weight_total_a);
    test_recall(2, weight_total_a);
}

void part_b()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part B", 8);
    for (blt::u32 i = 0; i < num_values; i++)
    {
        blt::generalized_matrix<float, 1, output_count> accum;
        for (blt::u32 k = 0; k < num_values; k++)
        {
            if (i == k)
                continue;
            accum += (outputs[k] * crosstalk(inputs[k].normalize(), inputs[i].normalize()));
        }
        crosstalk_values.assign_to_column_from_column_rows(accum, i);
    }
    for (blt::u32 i = 0; i < num_values; i++)
    {
        BLT_DEBUG_STREAM << crosstalk_values[i] << " Mag: " << crosstalk_values[i].magnitude() << "\n";
    }
}

int main()
{
    blt::logging::setLogOutputFormat("\033[94m[${{TIME}}]${{RC}} \033[35m(${{FILE}}:${{LINE}})${{RC}} ${{LF}}${{CNR}}${{STR}}${{RC}}\n");
    test_math();
    
    part_a();
    part_b();
}
