package com.a504.chatcomposer.global.exception;

import lombok.Getter;

@Getter
public class CustomException extends RuntimeException {
	private final CustomExceptionType exception;

	public CustomException(CustomExceptionType exception) {
		super(exception.getMessage());
		this.exception = exception;
	}
}
