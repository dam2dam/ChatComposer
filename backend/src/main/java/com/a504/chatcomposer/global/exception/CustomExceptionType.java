package com.a504.chatcomposer.global.exception;

import org.springframework.http.HttpStatus;

import lombok.Getter;

@Getter
public enum CustomExceptionType {

	RUNTIME_EXCEPTION(HttpStatus.BAD_REQUEST, "잘못된 요청입니다."),
	INTERNAL_SERVER_ERROR(HttpStatus.INTERNAL_SERVER_ERROR, "서버 오류 입니다."),

	// USER
	MEMBER_NOT_FOUND(HttpStatus.NOT_FOUND, "사용자 정보가 존재하지 않습니다."),

	// MUSIC
	MUSIC_NOT_FOUND(HttpStatus.NOT_FOUND, "음악 정보가 존재하지 않습니다."),
	DUPLICATE_FAVORITE_MUSIC(HttpStatus.BAD_REQUEST, "이미 좋아요 한 음악입니다."),
	FAIL_TO_DELETE_FAVORITE_MUSIC(HttpStatus.NOT_FOUND, "음악 좋아요 취소에 실패했습니다."),
	FAIL_TO_SAVE_COMPLETE_MUSIC(HttpStatus.BAD_REQUEST, "음악 게시에 실패했습니다."),
	TRANSFER_DATE_NOT_NUMBER(HttpStatus.BAD_REQUEST, "프롬프트 생성일은 숫자(문자열) 값으로 입력해주세요."),

	// TAG
	TAG_NOT_FOUND(HttpStatus.NOT_FOUND, "태그 전체 조회에 실패했습니다. 태그가 존재하지 않습니다."),
	DUPLICATE_TAG(HttpStatus.BAD_REQUEST, "이미 등록된 태그입니다."),
	UNREGISTERED_TAG(HttpStatus.NOT_FOUND, "태그 등록 후 사용해주세요"),

	// GENRE
	GENRE_NOT_FOUND(HttpStatus.NOT_FOUND, "존재하지 않는 장르입니다."),

	// BEAT
	BEAT_NOT_FOUND(HttpStatus.NOT_FOUND, "존재하지 않는 박자입니다. 박자는 60 ~ 180 사이의 값으로 입력해주세요."),
	BEAT_NOT_NUMBER(HttpStatus.BAD_REQUEST, "박자는 60 ~ 180 사이의 값으로 입력해주세요.");

	private final HttpStatus httpStatus;
	private final int code;
	private final String message;

	CustomExceptionType(HttpStatus httpStatus, String message) {
		this.httpStatus = httpStatus;
		this.code = httpStatus.value();
		this.message = message;
	}
}
