#pragma once
/*
 *  Copyright (C) 2024  Brett Terpstra
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef COSC_4P80_ASSIGNMENT_1_A1_H
#define COSC_4P80_ASSIGNMENT_1_A1_H

#include <blt/std/logging.h>
#include <blt/math/matrix.h>
#include <blt/math/log_util.h>
#include <algorithm>
#include <numeric>
#include <vector>

namespace a1
{
    template<blt::size_t rows, blt::size_t columns>
    using matrix_t = blt::generalized_matrix<float, rows, columns>;
    
    void test_math()
    {
        matrix_t<1, 4> input{1, -1, -1, 1};
        matrix_t<1, 3> output{1, 1, 1};
        matrix_t<4, 3> expected{
                blt::vec4{1, -1, -1, 1},
                blt::vec4{1, -1, -1, 1},
                blt::vec4{1, -1, -1, 1}
        };
        
        auto w_matrix = input.transpose() * output;
        BLT_ASSERT(w_matrix == expected && "MATH MATRIX FAILURE");
        
        blt::vec4 one{5, 1, 3, 0};
        blt::vec4 two{9, -5, -8, 3};
        
        matrix_t<1, 4> g1{5, 1, 3, 0};
        matrix_t<1, 4> g2{9, -5, -8, 3};
        
        BLT_ASSERT(g1 * g2.transpose() == blt::vec4::dot(one, two) && "MATH DOT FAILURE");
    }
    
    template<typename input_t>
    float crosstalk(const input_t& i, const input_t& j)
    {
        return i * j.transpose();
    }
    
    template<typename T>
    blt::size_t difference(const std::vector<T>& a, const std::vector<T>& b)
    {
        blt::size_t count = 0;
        for (const auto& [a_val, b_val] : blt::in_pairs(a, b))
        {
            if (a_val != b_val)
                count++;
        }
        return count;
    }
    
    template<typename T>
    bool equal(const std::vector<T>& a, const std::vector<T>& b)
    {
        return difference(a, b) == 0;
    }
    
//    template<typename weight_t, typename input_t, typename output_t>
//    std::pair<input_t, output_t> run_step(const weight_t& associated_weights, const input_t& input, const output_t& output)
//    {
//        output_t output_recall = input * associated_weights;
//        input_t input_recall = output_recall * associated_weights.transpose();
//
//        return std::pair{a1::threshold(input_recall, input), a1::threshold(output_recall, output)};
//    }
    
    template<typename weight_t, typename T, typename G>
    void check_recall(const weight_t& weights, const std::vector<G>& inputs, const std::vector<T>& outputs)
    {
        for (const auto& [index, val] : blt::enumerate(inputs))
        {
            auto result = run_step(weights, val, outputs[index]);
            if (result.first != val)
                BLT_ERROR("Recall of input #%ld failed", index + 1);
            else
                BLT_INFO("Recall of input #%ld passed", index + 1);
            if (result.second != outputs[index])
                BLT_ERROR("Recall of output #%ld failed", index + 1);
            else
                BLT_INFO("recall of output #%ld passed", index + 1);
        }
    }
}

#endif //COSC_4P80_ASSIGNMENT_1_A1_H
