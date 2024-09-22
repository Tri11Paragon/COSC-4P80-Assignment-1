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

#include <blt/math/matrix.h>
#include <blt/math/log_util.h>

namespace a1
{
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
        BLT_ASSERT(w_matrix == expected && "MATH MATRIX FAILURE");
        
        blt::vec4 one{5, 1, 3, 0};
        blt::vec4 two{9, -5, -8, 3};
        
        blt::generalized_matrix<float, 1, 4> g1{5, 1, 3, 0};
        blt::generalized_matrix<float, 1, 4> g2{9, -5, -8, 3};
        
        BLT_ASSERT(g1 * g2.transpose() == blt::vec4::dot(one, two) && "MATH DOT FAILURE");
    }
    
    template<typename input_t>
    float crosstalk(const input_t& i, const input_t& j)
    {
        return i * j.transpose();
    }
    
    template<typename T, blt::u32 rows, blt::u32 columns>
    blt::generalized_matrix<T, rows, columns> threshold(const blt::generalized_matrix<T, rows, columns>& y,
                                                        const blt::generalized_matrix<T, rows, columns>& base)
    {
        blt::generalized_matrix<T, rows, columns> result;
        for (blt::u32 i = 0; i < columns; i++)
        {
            for (blt::u32 j = 0; j < rows; j++)
                result[i][j] = y[i][j] > 1 ? 1 : (y[i][j] < -1 ? -1 : base[i][j]);
        }
        return result;
    }
    
    template<typename T, blt::size_t size>
    bool equal(const std::array<T, size>& a, const std::array<T, size>& b)
    {
        for (const auto& [index, val] : blt::enumerate(a))
        {
            if (b[index] != val)
                return false;
        }
        return true;
    }
    
    template<typename weight_t, typename input_t, typename output_t>
    std::pair<input_t, output_t> run_step(const weight_t& associated_weights, const input_t& input, const output_t& output)
    {
        output_t output_recall = input * associated_weights;
        input_t input_recall = output * associated_weights.transpose();
        
        return std::pair{a1::threshold(input_recall, input), a1::threshold(output_recall, output)};
    }
    
    template<typename weight_t, typename T, typename G, blt::size_t size>
    void check_recall(const weight_t& weights, const std::array<G, size>& inputs, const std::array<T, size>& outputs)
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
