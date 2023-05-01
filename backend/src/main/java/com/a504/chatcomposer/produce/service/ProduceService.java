package com.a504.chatcomposer.produce.service;

import com.a504.chatcomposer.produce.dto.request.MultipartFileReq;
import com.a504.chatcomposer.produce.dto.response.CoverUrlResp;

/**
 * Produce 관련 비즈니스 로직 처리를 위한 서비스 인터페이스 정의.
 */
public interface ProduceService {
    // 앨범 커버 이미지 저장
    String saveCover(MultipartFileReq multipartFileReq);

}
