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

#ifndef COSC_4P80_ASSIGNMENT_1_FWD_DECL_H
#define COSC_4P80_ASSIGNMENT_1_FWD_DECL_H

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
    
    enum class recall_error_t
    {
        // failed to predict input
        INPUT_FAILURE,
        // failed to predict output
        OUTPUT_FAILURE
    };
}

#endif //COSC_4P80_ASSIGNMENT_1_FWD_DECL_H
