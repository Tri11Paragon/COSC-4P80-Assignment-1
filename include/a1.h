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
    
    template<typename T, blt::u32 size>
    struct vec_formatter
    {
        public:
            explicit vec_formatter(const blt::vec<T, size>& data): data(data)
            {}
            
            template<typename Arr>
            std::string format(const Arr& has_index_changed)
            {
                using namespace blt::logging;
                std::string os;
                for (auto [index, value] : blt::enumerate(data))
                {
                    if (value >= 0)
                        os += ' ';
                    if (has_index_changed[index])
                        os += ansi::make_color(ansi::UNDERLINE);
                    os += blt::logging::to_string_stream(value);
                    if (has_index_changed[index])
                        os += ansi::make_color(ansi::RESET_UNDERLINE);
                    
                    if (index != size - 1)
                        os += ", ";
                }
                return os;
            }
        
        private:
            blt::vec<T, size> data;
    };
    
    template<typename T, blt::u32 size>
    vec_formatter(const blt::vec<T, size>& data) -> vec_formatter<T, size>;
}

#endif //COSC_4P80_ASSIGNMENT_1_A1_H
