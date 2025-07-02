// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MP_HELPER_H_
#define MP_HELPER_H_
#include <type_traits>
#include <utility>

namespace spconv {
template <class... T> struct mp_list {};

template <class T, T... I>
using mp_list_c = mp_list<std::integral_constant<T, I>...>;

namespace detail {

template <class... T, class F>
constexpr F mp_for_each_impl(mp_list<T...>, F &&f) {
  return std::initializer_list<int>{(f(T()), 0)...}, std::forward<F>(f);
}

template <class F> constexpr F mp_for_each_impl(mp_list<>, F &&f) {
  return std::forward<F>(f);
}

} // namespace detail

namespace detail {

template <class A, template <class...> class B> struct mp_rename_impl {
  // An error "no type named 'type'" here means that the first argument to
  // mp_rename is not a list
};

template <template <class...> class A, class... T, template <class...> class B>
struct mp_rename_impl<A<T...>, B> {
  using type = B<T...>;
};

} // namespace detail

template <class A, template <class...> class B>
using mp_rename = typename detail::mp_rename_impl<A, B>::type;

template <class L, class F> constexpr F mp_for_each(F &&f) {
  return detail::mp_for_each_impl(mp_rename<L, mp_list>(), std::forward<F>(f));
}
} // namespace spconv

#endif