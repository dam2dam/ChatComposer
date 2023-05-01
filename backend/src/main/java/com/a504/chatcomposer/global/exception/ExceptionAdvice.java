package com.a504.chatcomposer.global.exception;

import com.a504.chatcomposer.global.util.BaseResponseBody;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
@RequiredArgsConstructor
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
}
