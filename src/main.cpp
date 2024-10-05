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
            auto out = (input * weights);
            return {weights, (out * weights.transpose()).bipolar(), out.bipolar()};
        }
        
        [[nodiscard]] ping_pong pong() const
        {
            return run_step();
        }
        
        [[nodiscard]] const input_t& get_input() const
        {
            return input;
        }
        
        [[nodiscard]] const output_t& get_output() const
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
    
    private:
        weight_t weights;
        input_t input;
        output_t output;
};

class executor
{
    public:
        executor(const std::vector<input_t>& inputs, const std::vector<output_t>& outputs): weights(), inputs(inputs), outputs(outputs)
        {
            for (auto [in, out] : blt::in_pairs(inputs, outputs))
                weights += in.transpose() * out;
        }
        
        std::vector<output_t> crosstalk()
        {
            std::vector<output_t> crosstalk_data;
            crosstalk_data.resize(outputs.size());
            
            for (auto [i, c] : blt::enumerate(crosstalk_data))
            {
                for (auto [k, data] : blt::in_pairs(inputs, outputs).enumerate())
                {
                    if (i == k)
                        continue;
                    auto [a, b] = data;
                    
                    c += (b * (a.normalize() * inputs[i].normalize().transpose()));
                }
                
                BLT_TRACE("Input %ld has crosstalk magnitude: %f || %f", i, c.magnitude());
            }
            
            return crosstalk_data;
        }
        
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
        
        void print()
        {
            using namespace blt::logging;
            std::vector<std::string> input_lines;
            std::vector<std::string> output_lines;
            input_lines.resize(inputs.size());
            output_lines.resize(outputs.size());
            BLT_ASSERT(input_lines.size() == output_lines.size());
            
            for (auto [i, data] : blt::in_pairs(input_lines, output_lines).enumerate())
            {
                auto& [is, os] = data;
                
                auto ping_ping = steps.back()[i];
                
                auto& input_data = inputs[i];
                auto& output_data = outputs[i];
                
                ((((is += input_data == ping_ping.get_input() ? ansi::make_color(ansi::GREEN) : ansi::make_color(
                        ansi::RED)) += "[Input  ") += std::to_string(i)) += "]: ") += ansi::RESET;
                ((((os += output_data == ping_ping.get_output() ? ansi::make_color(ansi::GREEN) : ansi::make_color(
                        ansi::RED)) += "[Output ") += std::to_string(i)) += "]: ") += ansi::RESET;
            }
            
            for (auto [step_index, step_data] : blt::enumerate(steps))
            {
                for (auto [data_index, current_data] : blt::enumerate(step_data))
                {
                    auto current_input = current_data.get_input().vec_from_column_row();
                    auto current_output = current_data.get_output().vec_from_column_row();
                    
                    std::array<bool, decltype(current_input)::data_size> has_input_changed{};
                    std::array<bool, decltype(current_output)::data_size> has_output_changed{};
                    
                    if (step_index > 0)
                    {
                        auto& previous_data = steps[step_index - 1][data_index];
                        auto previous_input = previous_data.get_input().vec_from_column_row();
                        auto previous_output = previous_data.get_output().vec_from_column_row();
                        
                        for (auto [vec_index, data] : blt::zip(current_input, previous_input).enumerate())
                            has_input_changed[vec_index] = std::get<0>(data) != std::get<1>(data);
                        for (auto [vec_index, data] : blt::zip(current_output, previous_output).enumerate())
                            has_output_changed[vec_index] = std::get<0>(data) != std::get<1>(data);
                    }
                    
                    auto& is = input_lines[data_index];
                    auto& os = output_lines[data_index];
                    
                    ((is += "Vec") += blt::logging::to_string_stream(decltype(current_input)::data_size)) += "(";
                    ((os += "Vec") += blt::logging::to_string_stream(decltype(current_output)::data_size)) += "(";
                    
                    is += a1::vec_formatter(current_input).format(has_input_changed);
                    os += a1::vec_formatter(current_output).format(has_output_changed);
                    
                    is += ")";
                    os += ")";
                    
                    auto diff_o = (static_cast<blt::i32>(decltype(current_input)::data_size)
                                   - static_cast<blt::i32>(decltype(current_output)::data_size)) * 4;
                    auto diff_i = (static_cast<blt::i32>(decltype(current_output)::data_size)
                                   - static_cast<blt::i32>(decltype(current_input)::data_size)) * 4;
                    for (blt::i32 j = 0; j < diff_i; j++)
                        is += ' ';
                    for (blt::i32 j = 0; j < diff_o; j++)
                        os += ' ';
                    
                    if (step_index != steps.size() - 1)
                    {
                        is += " => ";
                        os += " => ";
                    } else
                    {
                        is += " || ";
                        os += " || ";
                    }
                }
            }
            
            for (auto [index, ping_ping] : blt::enumerate(steps.back()))
            {
                auto& is = input_lines[index];
                auto& os = output_lines[index];
                
                auto& input_data = inputs[index];
                auto& output_data = outputs[index];
                
                if (input_data != ping_ping.get_input())
                    ((is += ansi::make_color(ansi::RED)) += "[Failed]") += ansi::RESET;
                else
                    ((is += ansi::make_color(ansi::GREEN)) += "[Passed]") += ansi::RESET;
                
                if (output_data != ping_ping.get_output())
                    ((os += ansi::make_color(ansi::RED)) += "[Failed]") += ansi::RESET;
                else
                    ((os += ansi::make_color(ansi::GREEN)) += "[Passed]") += ansi::RESET;
            }
            
            BLT_TRACE("Changes between ping-pong steps are underlined.");
            for (auto [is, os] : blt::in_pairs(input_lines, output_lines))
            {
                BLT_TRACE_STREAM << is << "\n";
                BLT_TRACE_STREAM << os << "\n";
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

auto part_a_inputs = std::vector{input_1, input_2, input_3};
auto part_a_outputs = std::vector{output_1, output_2, output_3};

auto part_c_1_inputs = std::vector{input_1, input_2, input_3, input_4};
auto part_c_1_outputs = std::vector{output_1, output_2, output_3, output_4};

auto part_c_2_inputs = std::vector{input_1, input_2, input_3, input_4, input_5, input_6, input_7};
auto part_c_2_outputs = std::vector{output_1, output_2, output_3, output_4, output_5, output_6, output_7};

void part_a()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part A", 8);
    
    executor cute(part_a_inputs, part_a_outputs);
    cute.execute();
    cute.print();
}

void part_b()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part B", 8);
    executor cute(part_a_inputs, part_a_outputs);
    cute.crosstalk();
}

void part_c()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part C", 8);
    executor cute(part_c_1_inputs, part_c_1_outputs);
    cute.execute();
    cute.print();
    BLT_TRACE("--- { Part C with 3 extra pairs } ---");
    executor cute2(part_c_2_inputs, part_c_2_outputs);
    cute2.execute();
    cute2.print();
}

int main()
{
    blt::logging::setLogOutputFormat("\033[94m[${{TIME}}]${{RC}} \033[35m(${{FILE}}:${{LINE}})${{RC}} ${{LF}}${{CNR}}${{STR}}${{RC}}\n");
    a1::test_math();
    
    part_a();
    part_b();
    part_c();
    
    std::vector<input_t> test{input_t{1, -1, -1, -1, -1}, input_t{-1, 1, -1, -1, -1}, input_t{-1, -1, 1, -1, -1}};
    executor cute{test, part_a_outputs};
    cute.crosstalk();
}
