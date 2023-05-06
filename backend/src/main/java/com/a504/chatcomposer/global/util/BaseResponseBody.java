package com.a504.chatcomposer.global.util;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

import lombok.Getter;
import lombok.Setter;

/**
 * 서버 요청에대한 기본 응답값(바디) 정의.
 */
@Getter
@Setter
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class BaseResponseBody {
	String msg = null;
	Integer statusCode = null;

	public BaseResponseBody() {
	}

	public BaseResponseBody(Integer statusCode) {
		this.statusCode = statusCode;
	}

	public BaseResponseBody(Integer statusCode, String msg) {
		this.statusCode = statusCode;
		this.msg = msg;
	}

	public static BaseResponseBody of(Integer statusCode, String msg) {
		BaseResponseBody body = new BaseResponseBody();
		body.msg = msg;
		body.statusCode = statusCode;
		return body;
	}
}
