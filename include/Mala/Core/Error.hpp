/*
   Defined exceptions for programmer error.

   Shape/rank contract violations are bugs, so they throw.
   Recoverable failures will use std::expected later.
*/

#pragma once

#include <stdexcept>
#include <string>


namespace Mala
{

struct ShapeError : std::invalid_argument
{
    explicit ShapeError(std::string const& message)
        : std::invalid_argument(message)
    { }
};

} // Mala
