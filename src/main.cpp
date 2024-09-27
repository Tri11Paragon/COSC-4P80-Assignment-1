#include <iostream>
#include <blt/math/matrix.h>
#include <blt/math/log_util.h>
#include "blt/std/assert.h"
#include <blt/format/boxing.h>
#include <blt/std/iterator.h>
#include <a1.h>

constexpr blt::u32 num_values = 4;
constexpr blt::u32 input_count = 5;
constexpr blt::u32 output_count = 4;

using input_t = blt::generalized_matrix<float, 1, input_count>;
using output_t = blt::generalized_matrix<float, 1, output_count>;
using weight_t = decltype(std::declval<input_t>().transpose() * std::declval<output_t>());
using crosstalk_t = blt::generalized_matrix<float, output_count, num_values>;

// part a
input_t input_1{-1, 1, 1, 1, -1};
input_t input_2{-1, -1, -1, -1, 1};
input_t input_3{-1, -1, -1, 1, 1};
// part c 1
input_t input_4{1, 1, 1, 1, 1};
// part c 2
input_t input_5{-1, 1, -1, 1, 1};
input_t input_6{1, -1, 1, -1, 1};
input_t input_7{-1, 1, -1, 1, -1};

// part a
output_t output_1{1, 1, -1, 1};
output_t output_2{1, -1, -1, -1};
output_t output_3{-1, -1, 1, 1};
// part c 1
output_t output_4{-1, 1, 1, -1};
// part c 2
output_t output_5{1, 1, 1, 1};
output_t output_6{1, -1, -1, 1};
output_t output_7{1, 1, 1, -1};

const weight_t weight_1 = input_1.transpose() * output_1;
const weight_t weight_2 = input_2.transpose() * output_2;
const weight_t weight_3 = input_3.transpose() * output_3;
const weight_t weight_4 = input_4.transpose() * output_4;
const weight_t weight_5 = input_5.transpose() * output_5;
const weight_t weight_6 = input_6.transpose() * output_6;
const weight_t weight_7 = input_7.transpose() * output_7;

auto part_a_inputs = std::array{input_1, input_2, input_3};
auto part_a_outputs = std::array{output_1, output_2, output_3};

auto part_c_1_inputs = std::array{input_1, input_2, input_3, input_4};
auto part_c_1_outputs = std::array{output_1, output_2, output_3, output_4};

auto part_c_2_inputs = std::array{input_1, input_2, input_3, input_4, input_5, input_6, input_7};
auto part_c_2_outputs = std::array{output_1, output_2, output_3, output_4, output_5, output_6, output_7};

const auto weight_total_a = weight_1 + weight_2 + weight_3;
const auto weight_total_c = weight_total_a + weight_4;
const auto weight_total_c_2 = weight_total_c + weight_5 + weight_6 + weight_7;

crosstalk_t crosstalk_values{};

//template<typename Weights, typename Inputs, typename Outputs>
//void execute_BAM(const Weights& weights, const Inputs& input, const Outputs& output)
//{
//    auto current_inputs = input;
//    auto current_outputs = output;
//    auto next_inputs = current_inputs;
//    auto next_outputs = current_outputs;
//    blt::size_t iterations = 0;
//    constexpr blt::size_t max_iterations = 5;
//
//    do
//    {
//        current_inputs = next_inputs;
//        current_outputs = next_outputs;
//        ++iterations;
//        for (const auto& [index, val] : blt::enumerate(current_inputs))
//        {
//            auto next = a1::run_step(weights, val, current_outputs[index]);
//            next_inputs[index] = next.first;
//            next_outputs[index] = next.second;
//        }
//        // loop until no changes or we hit the iteration limit
//    } while ((!a1::equal(current_inputs, next_inputs) || !a1::equal(current_outputs, next_outputs)) && iterations < max_iterations);
//
//    BLT_DEBUG("Tracked after %ld iterations", iterations);
//    a1::check_recall(weights, next_inputs, next_outputs);
//}
//
//void part_a()
//{
//    blt::log_box_t box(BLT_TRACE_STREAM, "Part A", 8);
//
//    execute_BAM(weight_total_a, part_a_inputs, part_a_outputs);
//}
//
//void part_b()
//{
//    blt::log_box_t box(BLT_TRACE_STREAM, "Part B", 8);
//    for (blt::u32 i = 0; i < num_values; i++)
//    {
//        blt::generalized_matrix<float, 1, output_count> accum;
//        for (blt::u32 k = 0; k < num_values; k++)
//        {
//            if (i == k)
//                continue;
//            accum += (part_a_outputs[k] * a1::crosstalk(part_a_inputs[k].normalize(), part_a_inputs[i].normalize()));
//        }
//        crosstalk_values.assign_to_column_from_column_rows(accum, i);
//    }
//    for (blt::u32 i = 0; i < num_values; i++)
//    {
//        BLT_DEBUG_STREAM << crosstalk_values[i] << " Mag: " << crosstalk_values[i].magnitude() << "\n";
//    }
//}
//
//void part_c()
//{
//    blt::log_box_t box(BLT_TRACE_STREAM, "Part C", 8);
//    execute_BAM(weight_total_c, part_c_1_inputs, part_c_1_outputs);
//    BLT_TRACE("--- { Part C with 3 extra pairs } ---");
//    execute_BAM(weight_total_c_2, part_c_2_inputs, part_c_2_outputs);
//}

int main()
{
    blt::logging::setLogOutputFormat("\033[94m[${{TIME}}]${{RC}} \033[35m(${{FILE}}:${{LINE}})${{RC}} ${{LF}}${{CNR}}${{STR}}${{RC}}\n");
//    a1::test_math();
    
    for (const auto& [index, value] : blt::enumerate(part_c_2_inputs))
        BLT_TRACE_STREAM << index << " : " << value.vec_from_column_row() << '\n';
    
    BLT_TRACE("");
    
    for (const auto& [index, value] : blt::enumerate(part_c_2_inputs).rev())
        BLT_TRACE_STREAM << index << " : " << value.vec_from_column_row() << '\n';
    
    BLT_TRACE("");
    
    for (const auto& [index, value] : blt::enumerate(part_c_2_inputs).skip(3))
        BLT_TRACE_STREAM << index << " : " << value.vec_from_column_row() << '\n';
    
    BLT_TRACE("");
    
    for (const auto& [index, value] : blt::enumerate(part_c_2_inputs).skip(3).rev())
        BLT_TRACE_STREAM << index << " : " << value.vec_from_column_row() << '\n';
    
    BLT_TRACE("");
    
    for (const auto& [index, value] : blt::enumerate(part_c_2_inputs).rev().skip(3).rev())
        BLT_TRACE_STREAM << index << " : " << value.vec_from_column_row() << '\n';
    
    BLT_TRACE("");
    
    for (const auto& [index, value] : blt::enumerate(part_c_2_inputs).skip(2).take(3))
        BLT_TRACE_STREAM << index << " : " << value.vec_from_column_row() << '\n';
    
    for (const auto& [a, b] : blt::in_pairs(part_a_inputs, part_a_outputs))
    {
        BLT_TRACE_STREAM << a << " : " << b << "\n";
    }

//    BLT_TRACE("%s", blt::type_string<blt::meta::lowest_iterator_category<std::bidirectional_iterator_tag, std::random_access_iterator_tag, std::input_iterator_tag>::type>().c_str());


//    part_a();
//    part_b();
//    part_c();
}
