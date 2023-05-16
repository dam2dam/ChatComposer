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
public enum Beat {
	BPM60(0, 60, "60 beats per minute"),
	BPM70(1, 70, "70 beats per minute"),
	BPM80(2, 80, "80 beats per minute"),
	BPM90(3, 90, "90 beats per minute"),
	BPM100(4, 100, "100 beats per minute"),
	BPM110(5, 110, "110 beats per minute"),
	BPM120(6, 120, "120 beats per minute"),
	BPM130(7, 130, "130 beats per minute"),
	BPM140(8, 140, "140 beats per minute"),
	BPM150(9, 150, "150 beats per minute"),
	BPM160(10, 160, "160 beats per minute"),
	BPM170(11, 170, "170 beats per minute"),
	BPM180(12, 180, "180 beats per minute");

	private final int number;
	private final int bpmValue;
	private final String description;

	private static final Map<Integer, Beat> OPERATOR_MAP =
		Collections.unmodifiableMap(
			Stream.of(values()).collect(Collectors.toMap(Beat::getBpmValue, Function.identity()))
		);

	public static Beat findByBpmValue(int bpmValue) {
		if (OPERATOR_MAP.containsKey(bpmValue)) {
			return OPERATOR_MAP.get(bpmValue);
		} else {
			throw new CustomException(CustomExceptionType.BEAT_NOT_FOUND);
		}
	}

	public static Beat findByBpmValue(String bpmStringValue) {
		try {
			return findByBpmValue(Integer.parseInt(bpmStringValue));
		} catch (Exception exception) {
			throw new CustomException(CustomExceptionType.BEAT_NOT_NUMBER);
		}
	}

}
