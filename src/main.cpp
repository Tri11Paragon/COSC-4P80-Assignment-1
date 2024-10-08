#include <iostream>
#include <utility>
#include <blt/math/matrix.h>
#include <blt/math/log_util.h>
#include <blt/std/assert.h>
#include <blt/std/random.h>
#include <blt/format/boxing.h>
#include <blt/iterator/iterator.h>
#include <blt/parse/argparse.h>
#include <a1.h>

constexpr blt::u32 input_vec_size = 5;
constexpr blt::u32 output_vec_size = 4;

bool print_latex = false;

using input_t = a1::matrix_t<1, input_vec_size>;
using output_t = a1::matrix_t<1, output_vec_size>;
using weight_t = decltype(std::declval<input_t>().transpose() * std::declval<output_t>());

template<typename Os, typename T, blt::u32 size>
Os& print_vec_square(Os& o, const blt::vec<T, size>& v)
{
    o << "[";
    for (auto [i, f] : blt::enumerate(v))
    {
        o << f;
        if (i != size - 1)
            o << ", ";
    }
    o << "]";
    return o;
}

struct correctness_t
{
    blt::size_t correct_input = 0;
    blt::size_t correct_output = 0;
    blt::size_t incorrect_input = 0;
    blt::size_t incorrect_output = 0;
};

class ping_pong
{
    public:
        ping_pong(weight_t weights, input_t input, output_t output): weights(std::move(weights)), input(std::move(input)), output(std::move(output))
        {}
        
        [[nodiscard]] ping_pong run_step_from_inputs() const
        {
            auto out = (input * weights);
            return {weights, (out * weights.transpose()).bipolar(), out.bipolar()};
        }
        
        [[nodiscard]] ping_pong run_step_from_outputs() const
        {
            auto in = (output * weights.transpose());
            return {weights, in.bipolar(), (in * weights).bipolar()};
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
            generate_weights();
        }
        
        void add_pattern(input_t input, output_t output)
        {
            inputs.push_back(std::move(input));
            outputs.push_back(std::move(output));
        }
        
        void generate_weights()
        {
            for (auto [in, out] : blt::in_pairs(inputs, outputs))
                weights += in.transpose() * out;
        }
        
        void print_weights() const
        {
            BLT_TRACE_STREAM << "Weight Matrix: \n" << weights << "\n";
        }
        
        [[nodiscard]] std::vector<output_t> crosstalk() const
        {
            std::vector<output_t> crosstalk_data;
            crosstalk_data.resize(outputs.size());
            
            std::cout << "\\begin{tabular}{||c|c|c|c|c||}\n\\hline\n";
            for (auto [i, c] : blt::enumerate(crosstalk_data))
            {
                if (print_latex)
                {
                    if (i == 0)
                        std::cout << "Input " << i + 1 << " &   & Cos & Output Vector & Crosstalk Vector \\\\\n\\hline\\hline\n";
                    else
                        std::cout << "Input " << i + 1 << " & & & & \\\\\n\\hline\\hline\n";
                }
                for (auto [k, data] : blt::in_pairs(inputs, outputs).enumerate())
                {
                    if (i == k)
                        continue;
                    auto [a, b] = data;
                    
                    auto cos = a.normalize() * inputs[i].normalize().transpose();
                    auto this_talk = b * cos;
                    c += this_talk;
                    if (print_latex)
                    {
                        std::cout << "& Input " << k + 1 << " & " << cos << " & ";
                        print_vec_square(std::cout, b.vec_from_column_row());
                        std::cout << " & ";
                        print_vec_square(std::cout, this_talk.vec_from_column_row());
                        std::cout << "\\\\\n\\hline\n";
                    }
                }
                if (print_latex)
                {
                    std::cout << "\\hline\n\\multicolumn{5}{|c|}{Total Vector: ";
                    print_vec_square(std::cout, c.vec_from_column_row());
                    std::cout << "}\\\\\n\\multicolumn{5}{|c|}{Total Crosstalk: " << c.magnitude() << "} \\\\\n\\hline\\hline\n";
                }
            }
            std::cout << "\\end{tabular}\n";
            
            return crosstalk_data;
        }
        
        void print_crosstalk() const
        {
            auto talk = crosstalk();
            
            float total_talk = 0;
            
            for (auto [i, c] : blt::enumerate(talk))
            {
                BLT_TRACE_STREAM << "Input " << i << " [" << inputs[i].vec_from_column_row() << "] has crosstalk magnitude: " << c.magnitude()
                                 << "\n";
                total_talk += c.magnitude();
            }
            BLT_TRACE("Total Crosstalk: %f", total_talk);
        }
        
        void execute_input()
        {
            steps.clear();
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
                    next_pongs.emplace_back(ping.run_step_from_inputs());
                steps.emplace_back(std::move(next_pongs));
            } while (steps.rbegin()[0] != steps.rbegin()[1]);
        }
        
        void execute_output()
        {
            steps.clear();
            std::vector<ping_pong> initial_pings;
            initial_pings.reserve(outputs.size());
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
                    next_pongs.emplace_back(ping.run_step_from_outputs());
                steps.emplace_back(std::move(next_pongs));
            } while (steps.rbegin()[0] != steps.rbegin()[1]);
        }
        
        [[nodiscard]] input_t correct(const input_t& v) const
        {
            // outputs here do not matter.
            ping_pong current{weights, v, outputs.front()};
            ping_pong next{weights, v, outputs.front()};
            do
            {
                current = next;
                next = current.run_step_from_inputs();
                // run until stability
            } while (current != next);
            return next.get_input();
        }
        
        [[nodiscard]] correctness_t correctness() const
        {
            correctness_t results;
            for (auto [i, data] : blt::zip(inputs, outputs, steps.back()).enumerate())
            {
                auto& [input_data, output_data, ping_ping] = data;
                
                if (input_data == ping_ping.get_input())
                    results.correct_input++;
                else
                    results.incorrect_input++;
                
                if (output_data == ping_ping.get_output())
                    results.correct_output++;
                else
                    results.incorrect_output++;
            }
            return results;
        }
        
        void print_correctness() const
        {
            auto data = correctness();
            
            BLT_TRACE("Correct inputs  %ld Incorrect inputs  %ld | (%lf%%)", data.correct_input, data.incorrect_input,
                      static_cast<double>(data.correct_input * 100) / static_cast<double>(data.incorrect_input + data.correct_input));
            BLT_TRACE("Correct outputs %ld Incorrect outputs %ld | (%lf%%)", data.correct_output, data.incorrect_output,
                      static_cast<double>(data.correct_output * 100) / static_cast<double>(data.incorrect_output + data.correct_output));
            BLT_TRACE("Total correct   %ld Total incorrect   %ld | (%lf%%)", data.correct_input + data.correct_output,
                      data.incorrect_input + data.incorrect_output,
                      static_cast<double>(data.correct_input + data.correct_output) * 100 /
                      static_cast<double>(data.correct_input + data.correct_output + data.incorrect_input + data.incorrect_output));
        }
        
        void print_execution_results_latex_no_intermediates()
        {
            std::cout << "\\begin{longtable}{||";
            for (blt::size_t i = 0; i < 4; i++)
                std::cout << "c|";
            std::cout << "|}\n\t\\hline\n";
            std::cout << "\tType & Input Vectors & Result Vectors & Result\\\\\n\t\\hline\\hline\n";
            
            std::vector<std::string> input_lines;
            std::vector<std::string> output_lines;
            input_lines.resize(inputs.size());
            output_lines.resize(outputs.size());
            
            for (auto [i, is] : blt::enumerate(input_lines))
                (is += "Input ") += std::to_string(i + 1);
            for (auto [i, os] : blt::enumerate(output_lines))
                (os += "Output ") += std::to_string(i + 1);
            
            for (const auto& [step_idx, step] : blt::enumerate(steps))
            {
                for (auto [idx, pong] : blt::enumerate(step))
                {
                    if (!(step_idx == 0 || step_idx == steps.size()-1))
                        break;
                    auto& is = input_lines[idx];
                    auto& os = output_lines[idx];
                    
                    is += " & ";
                    os += " & ";
                    
                    std::stringstream stream;
                    print_vec_square(stream, pong.get_input().vec_from_column_row());
                    is += stream.str();
                    stream = {};
                    print_vec_square(stream, pong.get_output().vec_from_column_row());
                    os += stream.str();
                }
            }
            auto result = steps.back();
            for (auto [idx, ping] : blt::zip(result, inputs, outputs).enumerate())
            {
                auto [pong, input, output] = ping;
                auto& is = input_lines[idx];
                auto& os = output_lines[idx];
                
                is += " & ";
                os += " & ";
                
                if (pong.get_input() == input)
                    is += "Correct";
                else
                    is += "Incorrect";
                if (pong.get_output() == output)
                    os += "Correct";
                else
                    os += "Incorrect";
            }
            for (auto [is, os] : blt::in_pairs(input_lines, output_lines))
            {
                std::cout << '\t' << is << " \\\\\n\t\\hline\n";
                std::cout << '\t' << os << " \\\\\n\t\\hline\n";
            }
            std::cout << "\t\\caption{}\n";
            std::cout << "\t\\label{tbl:}\n";
            std::cout << "\\end{longtable}\n";
        }
        
        void print_execution_results_latex()
        {
            std::cout << "\\begin{longtable}{||";
            for (blt::size_t i = 0; i < steps.size() + 3; i++)
                std::cout << "c|";
            std::cout << "|}\n\t\\hline\n";
            std::cout << "\tType & Input Vectors & \\multicolumn{" << steps.size() - 2
                      << "}{|c|}{Intermediate Vectors} & Result Vectors & Result\\\\\n\t\\hline\\hline\n";
            
            std::vector<std::string> input_lines;
            std::vector<std::string> output_lines;
            input_lines.resize(inputs.size());
            output_lines.resize(outputs.size());
            
            for (auto [i, is] : blt::enumerate(input_lines))
                (is += "Input ") += std::to_string(i + 1);
            for (auto [i, os] : blt::enumerate(output_lines))
                (os += "Output ") += std::to_string(i + 1);
            
            for (const auto& step : steps)
            {
                for (auto [idx, pong] : blt::enumerate(step))
                {
                    auto& is = input_lines[idx];
                    auto& os = output_lines[idx];
                    
                    is += " & ";
                    os += " & ";
                    
                    std::stringstream stream;
                    print_vec_square(stream, pong.get_input().vec_from_column_row());
                    is += stream.str();
                    stream = {};
                    print_vec_square(stream, pong.get_output().vec_from_column_row());
                    os += stream.str();
                }
            }
            auto result = steps.back();
            for (auto [idx, ping] : blt::zip(result, inputs, outputs).enumerate())
            {
                auto [pong, input, output] = ping;
                auto& is = input_lines[idx];
                auto& os = output_lines[idx];
                
                is += " & ";
                os += " & ";
                
                if (pong.get_input() == input)
                    is += "Correct";
                else
                    is += "Incorrect";
                if (pong.get_output() == output)
                    os += "Correct";
                else
                    os += "Incorrect";
            }
            for (auto [is, os] : blt::in_pairs(input_lines, output_lines))
            {
                std::cout << '\t' << is << " \\\\\n\t\\hline\n";
                std::cout << '\t' << os << " \\\\\n\t\\hline\n";
            }
            std::cout << "\t\\caption{}\n";
            std::cout << "\t\\label{tbl:}\n";
            std::cout << "\\end{longtable}\n";
        }
        
        void print_execution_results()
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
        
        std::vector<ping_pong>& get_results()
        {
            return steps.back();
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
    cute.print_weights();
    std::cout << "\n";
    
    BLT_TRACE("Running from inputs:");
    cute.execute_input();
    cute.print_execution_results();
    cute.print_correctness();
    if (print_latex)
        cute.print_execution_results_latex();
    
    std::cout << "\n";
    BLT_TRACE("Running from outputs:");
    cute.execute_output();
    cute.print_execution_results();
    cute.print_correctness();
    if (print_latex)
        cute.print_execution_results_latex();
}

void part_b()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part B", 8);
    executor cute(part_a_inputs, part_a_outputs);
    cute.print_crosstalk();
//    cute.print_crosstalk_table();
}

void part_c()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part C", 8);
    executor cute(part_c_1_inputs, part_c_1_outputs);
    cute.execute_input();
    cute.print_execution_results();
    cute.print_correctness();
    cute.print_crosstalk();
    if (print_latex)
        cute.print_execution_results_latex_no_intermediates();
    cute.execute_output();
    cute.print_execution_results();
    cute.print_correctness();
    if (print_latex)
        cute.print_execution_results_latex_no_intermediates();
    BLT_TRACE("--- { Part C with 3 extra pairs } ---");
    executor cute2(part_c_2_inputs, part_c_2_outputs);
    cute2.execute_input();
    cute2.print_execution_results();
    cute2.print_correctness();
    cute2.print_crosstalk();
    if (print_latex)
        cute2.print_execution_results_latex_no_intermediates();
    cute2.execute_output();
    cute2.print_execution_results();
    cute2.print_correctness();
    if (print_latex)
        cute2.print_execution_results_latex_no_intermediates();
}

blt::size_t hdist(const input_t& a, const input_t& b)
{
    blt::size_t diff = 0;
    for (auto [av, bv] : blt::in_pairs(a.vec_from_column_row(), b.vec_from_column_row()))
        diff += (av != bv ? 1 : 0);
    return diff;
}

void part_d()
{
    blt::log_box_t box(BLT_TRACE_STREAM, "Part D", 8);
    blt::random::random_t random(std::random_device{}());
    executor cute(part_a_inputs, part_a_outputs);
    constexpr blt::size_t number_of_runs = 80;
    std::vector<blt::size_t> mutations;
    std::vector<blt::size_t> corrections;
    blt::size_t total_corrections = 0;
    blt::size_t total_mutations = 0;
    blt::size_t min_corrections = std::numeric_limits<blt::size_t>::max();
    blt::size_t max_corrections = 0;
    blt::size_t min_mutations = std::numeric_limits<blt::size_t>::max();
    blt::size_t max_mutations = 0;
    for (blt::size_t run = 0; run < number_of_runs; run++)
    {
        auto pos = random.get_size_t(0, part_a_inputs.size());
        auto original = part_a_inputs[pos];
        auto modified = original;
        for (blt::size_t i = 0; i < std::remove_reference_t<decltype(modified)>::data_columns; i++)
        {
            if (random.choice(0.2))
            {
                // flip value of this location
                auto& d = modified[i][0];
                if (d >= 0)
                    d = -1;
                else
                    d = 1;
            }
        }
        auto corrected = cute.correct(modified);
        
        auto dist_o_m = hdist(original, modified);
        auto dist_o_c = hdist(original, corrected);
        
        corrections.push_back(dist_o_c);
        mutations.push_back(dist_o_m);
        total_corrections += dist_o_c;
        total_mutations += dist_o_m;
        
        min_corrections = std::min(dist_o_c, min_corrections);
        max_corrections = std::max(dist_o_c, max_corrections);
        
        min_mutations = std::min(dist_o_m, min_mutations);
        max_mutations = std::max(dist_o_m, max_mutations);
        
        if (print_latex)
        {
            std::cout << run + 1 << " & ";
            print_vec_square(std::cout, original.vec_from_column_row()) << " & ";
            print_vec_square(std::cout, modified.vec_from_column_row()) << " & ";
            std::cout << dist_o_m << " & ";
            print_vec_square(std::cout, corrected.vec_from_column_row()) << " & ";
            std::cout << dist_o_c << " \\\\ \n\\hline\n";
        } else
        {
            BLT_TRACE_STREAM << "Run " << run << " " << original.vec_from_column_row() << " || mutated " << modified.vec_from_column_row()
                             << " difference: " << dist_o_m << " || corrected " << corrected.vec_from_column_row() << " || difference: " << dist_o_c
                             << "\n";
        }
    }
    double mean_corrections = static_cast<double>(total_corrections) / number_of_runs;
    double mean_mutations = static_cast<double>(total_mutations) / number_of_runs;
    
    double stddev_corrections = 0;
    double stddev_mutations = 0;
    
    for (const auto& v : corrections)
    {
        auto x = (static_cast<double>(v) - mean_corrections);
        stddev_corrections += x * x;
    }
    
    for (const auto& v : mutations)
    {
        auto x = (static_cast<double>(v) - mean_mutations);
        stddev_mutations += x * x;
    }
    
    stddev_corrections /= number_of_runs;
    stddev_mutations /= number_of_runs;
    
    stddev_corrections = std::sqrt(stddev_corrections);
    stddev_mutations = std::sqrt(stddev_mutations);
    
    std::cout << "Mean Distance Corrections: " << mean_corrections << " Stddev: " << stddev_corrections << " Min: " << min_corrections << " Max: "
              << max_corrections << '\n';
    std::cout << "Mean Distance Mutations: " << mean_mutations << " Stddev: " << stddev_mutations << " Min: " << min_mutations << " Max: "
              << max_mutations << '\n';
}

int main(int argc, const char** argv)
{
    blt::arg_parse parser;
    parser.addArgument(blt::arg_builder{"--latex", "-l"}.setAction(blt::arg_action_t::STORE_TRUE).setDefault(false).build());
    
    auto args = parser.parse_args(argc, argv);
    print_latex = blt::arg_parse::get<bool>(args["latex"]);
    
    blt::logging::setLogOutputFormat("\033[94m[${{TIME}}]${{RC}} \033[35m(${{FILE}}:${{LINE}})${{RC}} ${{LF}}${{CNR}}${{STR}}${{RC}}\n");
    a1::test_math();
    
    part_a();
    part_b();
    part_c();
    part_d();
    
    std::vector<input_t> test{input_t{1, -1, -1, -1, -1}, input_t{-1, 1, -1, -1, -1}, input_t{-1, -1, 1, -1, -1}};
    executor cute{test, part_a_outputs};
    cute.print_crosstalk();
}
