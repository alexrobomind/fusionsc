#pragma once

#include "variant.h"

namespace gold_fish
{
	struct nullopt_t {};
	constexpr nullopt_t nullopt{};

	template <class T> struct enable_if_exists { using type = void; };
	template <class T> using enable_if_exists_t = typename enable_if_exists<T>::type;

	template <class T, bool complex> class optional_with_invalid_base
	{
	protected:
		using invalid_state = typename T::invalid_state;
	public:
		optional_with_invalid_base() = default;
		optional_with_invalid_base(const optional_with_invalid_base& rhs)
		{
			if (invalid_state::is(rhs.m_data))
				invalid_state::set(m_data);
			else
				new (&m_data) T(reinterpret_cast<T&>(rhs.m_data));
		}
		optional_with_invalid_base(optional_with_invalid_base&& rhs)
		{
			if (invalid_state::is(rhs.m_data))
				invalid_state::set(m_data);
			else
				new (&m_data) T(std::move(reinterpret_cast<T&>(rhs.m_data)));
		}
		~optional_with_invalid_base()
		{
			if (!invalid_state::is(m_data))
				reinterpret_cast<T&>(m_data).~T();
		}

	protected:
		std::aligned_storage_t<sizeof(T), alignof(T)> m_data;
	};
	template <class T> class optional_with_invalid_base<T, false /*complex*/>
	{
	protected:
		using invalid_state = typename T::invalid_state;
		std::aligned_storage_t<sizeof(T), alignof(T)> m_data;
	};

	template <class T, class dummy = void>
	class optional
	{
	public:
		optional() = default;
		optional(const optional&) = default;
		optional(optional&&) = default;
		optional& operator = (const optional&) = default;
		optional& operator = (optional&&) = default;

		optional(nullopt_t) {}
		optional(const T& t) : m_data(t) {}
		optional(T&& t) : m_data(std::move(t)) {}
		optional& operator = (nullopt_t) { m_data = nullopt; return *this; }
		optional& operator = (const T& t) { m_data = t; return *this; }
		optional& operator = (T&& t) { m_data = std::move(t); return *this; }

		T& operator*() & { return m_data.as<T>(); }
		const T& operator*() const & { return m_data.as<T>(); }
		T&& operator*() && { return std::move(m_data.as<T>()); }

		T* operator ->() { return &m_data.as<T>(); }
		const T* operator ->() const { return &m_data.as<T>(); }
		explicit operator bool() const { return m_data.is<T>(); }

		friend bool operator == (const optional& lhs, nullopt_t) { return lhs.m_data.is<nullopt_t>(); }
		friend bool operator == (const optional& lhs, const T& t) { return lhs.m_data.is<T>() && lhs.m_data.as<T>() == t; }
		friend bool operator == (const optional& lhs, const optional& rhs) { return lhs.m_data == rhs.m_data; }

		friend bool operator != (const optional& lhs, nullopt_t) { return !lhs.m_data.is<nullopt_t>(); }
		friend bool operator != (const optional& lhs, const T& t) { return !lhs.m_data.is<T>() || lhs.m_data.as<T>() != t; }
		friend bool operator != (const optional& lhs, const optional& rhs) { return lhs.m_data != rhs.m_data; }

	private:
		variant<nullopt_t, T> m_data;
	};


	template <class T> class optional<T, enable_if_exists_t<typename T::invalid_state>> :
		public optional_with_invalid_base<T,
			std::is_trivially_copy_constructible<T>::value &&
			std::is_trivially_move_constructible<T>::value &&
			std::is_trivially_destructible<T>::value>
	{
	public:
		optional()
		{
			invalid_state::set(m_data);
		}
		optional(const optional&) = default;
		optional(optional&&) = default;

		optional& operator = (const optional& rhs)
		{
			if (rhs)
				return *this = rhs;
			else
				return *this = nullopt;
		}
		optional& operator = (optional&& rhs)
		{
			if (rhs)
				return *this = std::move(rhs);
			else
				return *this = nullopt;
		}
		
		optional(nullopt_t) : optional() {}
		optional(const T& t)
		{
			if (std::is_trivially_copy_constructible<T>::value)
				memcpy(&m_data, &t, sizeof(T));
			else
				new (&m_data) T(t);
			assert(*this);
		}
		optional(T&& t)
		{
			if (std::is_trivially_copy_constructible<T>::value)
				memcpy(&m_data, &t, sizeof(T));
			else
				new (&m_data) T(std::move(t));
			assert(*this);
		}
		optional& operator = (nullopt_t)
		{
			if (*this)
				(*this)->~T();
			invalid_state::set(m_data);
			assert(!*this);
			return *this;
		}
		optional& operator = (const T& t)
		{
			if (*this)
				return *this = t;

			if (std::is_trivially_copy_constructible<T>::value)
				memcpy(&m_data, &data, sizeof(T));
			else
				new (&m_data) T(data);
			assert(*this);
			return *this;
		}
		optional& operator = (T&& t)
		{
			if (*this)
				return *this = std::move(t);

			if (std::is_trivially_copy_constructible<T>::value)
				memcpy(&m_data, &data, sizeof(T));
			else
				new (&m_data) T(std::move(data));
			assert(*this);
			return *this;
		}

		T& operator*() & { assert(*this); return reinterpret_cast<T&>(m_data); }
		const T& operator*() const & { assert(*this); return reinterpret_cast<const T&>(m_data); }
		T&& operator*() && { assert(*this); return std::move(reinterpret_cast<T&>(m_data)); }

		T* operator ->() { return &reinterpret_cast<T&>(m_data); }
		const T* operator ->() const { return &reinterpret_cast<const T&>(m_data); }
		explicit operator bool() const { return !invalid_state::is(m_data); }

		friend bool operator == (const optional& lhs, nullopt_t) { return static_cast<bool>(lhs); }
		friend bool operator == (const optional& lhs, const T& t) { return lhs && *lhs == t; }
		friend bool operator == (const optional& lhs, const optional& rhs)
		{
			if (rhs)
				return lhs == *rhs;
			else
				return lhs == nullopt;
		}

		friend bool operator != (const optional& lhs, nullopt_t) { return !(lhs == nullopt); }
		friend bool operator != (const optional& lhs, const T& t) { return !(lhs == t); }
		friend bool operator != (const optional& lhs, const optional& rhs) { return !(lhs == rhs); }
	};
}