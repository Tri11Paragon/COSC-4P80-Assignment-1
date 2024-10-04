#include <iostream>
#include <utility>
#include <blt/math/matrix.h>
#include <blt/math/log_util.h>
#include "blt/std/assert.h"
#include <blt/format/boxing.h>
#include <blt/iterator/iterator.h>
#include <a1.h>

constexpr blt::u32 num_values_part_a = 3;
constexpr blt::u32 num_values_part_c1 = 4;
constexpr blt::u32 num_values_part_c2 = 7;
constexpr blt::u32 input_vec_size = 5;
constexpr blt::u32 output_vec_size = 4;

using input_t = a1::matrix_t<1, input_vec_size>;
using output_t = a1::matrix_t<1, output_vec_size>;
using weight_t = decltype(std::declval<input_t>().transpose() * std::declval<output_t>());
using crosstalk_t = a1::matrix_t<1, output_vec_size>;

class ping_pong
{
    public:
        ping_pong(weight_t weights, input_t input): weights(std::move(weights)), input(std::move(input))
        {}
        
        ping_pong(weight_t weights, input_t input, output_t output): weights(std::move(weights)), input(std::move(input)), output(std::move(output))
        {}
        
        [[nodiscard]] ping_pong run_step() const
        {
            auto out = input * weights;
            return {weights, threshold(out * weights.transpose()), threshold(out)};
        }
        
        [[nodiscard]] ping_pong pong() const
        {
            return run_step();
        }
        
        input_t get_input()
        {
            return input;
        }
        
        output_t get_output()
        {
            return output;
        }
        
        friend bool operator==(const ping_pong& a, const ping_pong& b)
        {
            return a.input == b.input && a.output == b.output;
        }
        
        friend bool operator!=(const ping_pong& a, const ping_pong& b)
        {
            return a.input != b.input || a.output != b.output;
        }
        
        template<blt::u32 rows, blt::u32 columns>
        static a1::matrix_t<rows, columns> threshold(const a1::matrix_t<rows, columns>& y)
        {
            a1::matrix_t<rows, columns> result;
            for (blt::u32 i = 0; i < columns; i++)
            {
                for (blt::u32 j = 0; j < rows; j++)
                    result[i][j] = y[i][j] >= 0 ? 1 : -1;
            }
            return result;
        }
    
    private:
        weight_t weights;
        input_t input;
        output_t output;
};

class executor
{
    public:
        executor(weight_t weights, std::vector<input_t> inputs, std::vector<output_t> outputs):
                weights(std::move(weights)), inputs(std::move(inputs)), outputs(std::move(outputs))
        {}
        
        void execute()
        {
            std::vector<ping_pong> initial_pings;
            initial_pings.reserve(inputs.size());
            for (auto [input, output] : blt::in_pairs(inputs, outputs))
                initial_pings.emplace_back(weights, input, output);
            steps.emplace_back(std::move(initial_pings));
            // execute while the entries don't equal each other (no stability in the system)
            do
            {
                auto& prev = steps.rbegin()[0];
                std::vector<ping_pong> next_pongs;
                next_pongs.reserve(prev.size());
                for (auto& ping : prev)
                    next_pongs.emplace_back(ping.pong());
                steps.emplace_back(std::move(next_pongs));
            } while (steps.rbegin()[0] != steps.rbegin()[1]);
        }
        
        void print_chains()
        {
            using namespace blt::logging;
            std::vector<std::string> input;
            input.reserve(inputs.size());
            input.resize(inputs.size());
            std::vector<std::string> output;
            output.reserve(outputs.size());
            output.resize(outputs.size());
            for (auto [index, ref_pair] : blt::in_pairs(input, output).enumerate())
            {
                auto& [i, o] = ref_pair;
                (((i += "[Input  ") += std::to_string(index)) += "]: ") += ansi::RESET;
                (((o += "[Output ") += std::to_string(index)) += "]: ") += ansi::RESET;
                auto& current_line = steps[index];
                for (auto [pos, ping] : blt::enumerate(current_line))
                {
                    auto input_vec = ping.get_input().vec_from_column_row();
                    auto output_vec = ping.get_output().vec_from_column_row();
                    auto input_size = input_vec.end() - input_vec.begin();
                    auto output_size = output_vec.end() - output_vec.begin();
                    if (pos > 0)
                    {
                        ((i += "Vec") += blt::logging::to_string_stream(input_size)) += "(";
                        ((o += "Vec") += blt::logging::to_string_stream(output_size)) += "(";
                        auto prev_input_vec = current_line[pos - 1].get_input().vec_from_column_row();
                        auto prev_output_vec = current_line[pos - 1].get_output().vec_from_column_row();
                        for (auto [j, prev_value_t] : blt::zip(input_vec, prev_input_vec).enumerate())
                        {
                            auto [cur_input, prev_input] = prev_value_t;
                            
                            if (cur_input > 0)
                                i += ' ';
                            
                            if (cur_input != prev_input)
                                i += ansi::make_color(ansi::UNDERLINE);
                            i += blt::logging::to_string_stream(cur_input);
                            if (cur_input != prev_input)
                                i += ansi::make_color(ansi::RESET_UNDERLINE);
                            
                            if (static_cast<blt::ptrdiff_t>(j) != input_size - 1)
                                i += ", ";
                        }
                        for (auto [j, prev_value_t] : blt::zip(output_vec, prev_output_vec).enumerate())
                        {
                            auto [cur_output, prev_output] = prev_value_t;
                            
                            if (cur_output > 0)
                                o += ' ';
                            if (cur_output != prev_output)
                                o += ansi::make_color(ansi::UNDERLINE);
                            o += blt::logging::to_string_stream(cur_output);
                            if (cur_output != prev_output)
                                o += ansi::make_color(ansi::RESET_UNDERLINE);
                            
                            if (static_cast<blt::ptrdiff_t>(j) != output_size - 1)
                                o += ", ";
                        }
                        i += ")";
                        o += ")";
                    } else
                    {
                        i += blt::logging::to_string_stream(input_vec);
                        o += blt::logging::to_string_stream(output_vec);
                    }
                    
                    auto diff_o = (static_cast<blt::ptrdiff_t>(input_size) - static_cast<blt::ptrdiff_t>(output_size)) * 4;
                    auto diff_i = (static_cast<blt::ptrdiff_t>(output_size) - static_cast<blt::ptrdiff_t>(input_size)) * 4;
                    for (blt::ptrdiff_t j = 0; j < diff_i; j++)
                        i += ' ';
                    for (blt::ptrdiff_t j = 0; j < diff_o; j++)
                        o += ' ';
                    
                    if (pos != current_line.size() - 1)
                    {
                        i += " => ";
                        o += " => ";
                    }
                }
            }
            
            for (auto [i, o] : blt::in_pairs(input, output))
            {
                BLT_TRACE_STREAM << i << "\n";
                BLT_TRACE_STREAM << o << "\n";
            }
        }
    
    private:
        weight_t weights;
        std::vector<input_t> inputs;
        std::vector<output_t> outputs;
        std::vector<std::vector<ping_pong>> steps;
};

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

auto part_a_inputs = std::vector{input_1, input_2, input_3};
auto part_a_outputs = std::vector{output_1, output_2, output_3};

auto part_c_1_inputs = std::vector{input_1, input_2, input_3, input_4};
auto part_c_1_outputs = std::vector{output_1, output_2, output_3, output_4};

auto part_c_2_inputs = std::vector{input_1, input_2, input_3, input_4, input_5, input_6, input_7};
auto part_c_2_outputs = std::vector{output_1, output_2, output_3, output_4, output_5, output_6, output_7};

const auto weight_total_a = weight_1 + weight_2 + weight_3;
const auto weight_total_c = weight_total_a + weight_4;
const auto weight_total_c_2 = weight_total_c + weight_5 + weight_6 + weight_7;

crosstalk_t crosstalk_values[num_values_part_a];

void part_a()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part A", 8);
    
    executor cute(weight_total_a, part_a_inputs, part_a_outputs);
    cute.execute();
    cute.print_chains();
}

void part_b()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part B", 8);
    for (blt::u32 i = 0; i < num_values_part_a; i++)
    {
        crosstalk_values[i] = {};
        for (blt::u32 k = 0; k < num_values_part_a; k++)
        {
            if (i == k)
                continue;
            crosstalk_values[i] += (part_a_outputs[i] * a1::crosstalk(part_a_inputs[i].normalize(), part_a_inputs[k].normalize())).abs();
        }
    }
    for (const auto& crosstalk_value : crosstalk_values)
    {
        auto vec = crosstalk_value.vec_from_column_row();
        BLT_DEBUG_STREAM << vec << " Mag: " << vec.magnitude() << "\n";
    }
}

void part_c()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part C", 8);
//    execute_BAM(weight_total_c, part_c_1_inputs, part_c_1_outputs);
    BLT_TRACE("--- { Part C with 3 extra pairs } ---");
//    execute_BAM(weight_total_c_2, part_c_2_inputs, part_c_2_outputs);
}

int main()
{
    blt::logging::setLogOutputFormat("\033[94m[${{TIME}}]${{RC}} \033[35m(${{FILE}}:${{LINE}})${{RC}} ${{LF}}${{CNR}}${{STR}}${{RC}}\n");
    a1::test_math();
    
    part_a();
    part_b();
    part_c();
}
