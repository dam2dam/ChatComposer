package com.a504.chatcomposer.produce.dto.request;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * MultipartFile을 저장하기 위한 API ([POST] /produce/cover) 요청에 필요한 리퀘스트 바디 정의.
 */
@Data
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@Schema(title = "파일 Vo")
public class MultipartFileReq {
    @Schema(title = "멀티파트 이미지 File", description = "멀티 파트 이미지 File")
    private MultipartFile image;
}