package com.a504.chatcomposer.global.exception;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import com.a504.chatcomposer.global.util.BaseResponseBody;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@RestControllerAdvice    // 모든 @Controller에 대한, 전역적으로 발생할 수 있는 예외를 잡음
@RequiredArgsConstructor
@Slf4j
public class ExceptionAdvice {

	/*
	 * 사용 현황
	 * ProduceServiceImpl : createCover(앨범커버 파일 제작 실패 시), saveCover(앨범커버 파일 저장 실패 시)
	 */
	@ExceptionHandler(IllegalArgumentException.class)
	@ResponseStatus(HttpStatus.NOT_FOUND)
	protected ResponseEntity<? extends BaseResponseBody> applicantDuplicationException(IllegalArgumentException e) {
		System.out.println(e.getMessage());
		return ResponseEntity.status(HttpStatus.NOT_FOUND).body(BaseResponseBody.of(404, e.getMessage()));
	}

	/**
	 * Exception이 발생하면 호출되는 예외 핸들러
	 * @return (500 Internal Server Error)를 담은 ResponseEntity
	 */
	@ExceptionHandler(value = Exception.class)
	public ResponseEntity<? extends BaseResponseBody> exceptionHandler(Exception exception) {
		log.info(exception.getMessage());
		return getResponseEntity(CustomExceptionType.INTERNAL_SERVER_ERROR);
	}

	/**
	 * RuntimeException이 발생하면 호출되는 런타임 예외 핸들러
	 * @return (400 Bad Request)를 담은 ResponseEntity
	 */
	@ExceptionHandler(value = RuntimeException.class)
	public ResponseEntity<? extends BaseResponseBody> runtimeExceptionHandler(RuntimeException runtimeException) {
		log.info(runtimeException.getMessage());
		return getResponseEntity(CustomExceptionType.RUNTIME_EXCEPTION);
	}

	/**
	 * CustomException이 발생하면 호출되는 커스텀 예외 핸들러
	 * @return 각 예외의 상태, 코드, 메시지를 담은 ResponseEntity
	 */
	@ExceptionHandler(value = CustomException.class)
	public ResponseEntity<? extends BaseResponseBody> customExceptionHandler(CustomException customException) {
		return getResponseEntity(customException.getException());
	}

	/**
	 * 상태, 코드, 메시지를 담아 ResponseEntity 생성
	 */
	private ResponseEntity<BaseResponseBody> getResponseEntity(CustomExceptionType customExceptionType) {
		return ResponseEntity
			.status(customExceptionType.getHttpStatus())
			.body(BaseResponseBody.of(
				customExceptionType.getCode(),
				customExceptionType.getMessage()
			));
	}
}
