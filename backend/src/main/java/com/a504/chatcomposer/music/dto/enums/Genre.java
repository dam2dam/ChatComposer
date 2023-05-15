package com.a504.chatcomposer.music.dto.enums;

import java.util.Collections;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.a504.chatcomposer.global.exception.CustomException;
import com.a504.chatcomposer.global.exception.CustomExceptionType;

import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter
@AllArgsConstructor
public enum Genre {

	// 임시 장르 (수정 예정)
	POP(0),
	HIP_HOP(1),
	ROCK(2),
	SOUL(3),
	REGGAE(4),
	COUNTRY(5),
	FUNK(6),
	FOLK(7),
	MIDDLE_EASTERN(8),
	JAZZ(9),
	DISCO(10),
	CLASSICAL(11),
	ELECTRONIC(12),
	BLUES(13),
	NEW_AGE(14),
	VOCAL(15),
	CHRISTIAN(16),
	SKA(17),
	TRADITIONAL(18),
	INDEPENDENT(19);

	/**
	 * 장르 고유 번호
	 */
	private final int number;

	private static final Map<Integer, Genre> OPERATOR_MAP =
		Collections.unmodifiableMap(
			Stream.of(values()).collect(Collectors.toMap(Genre::getNumber, Function.identity())));

	public static Genre findByNumber(int number) {
		if (OPERATOR_MAP.containsKey(number)) {
			return OPERATOR_MAP.get(number);
		} else {
			throw new CustomException(CustomExceptionType.GENRE_NOT_FOUND);
		}
	}
}
