package com.a504.chatcomposer.produce.dto.request;

import lombok.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * 원본 음악을 저장하기 위한 API ([POST] /produce/cover) 요청에 필요한 리퀘스트 바디 정의.
 */
@Data
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
public class OriginalMusicReq {
    private MultipartFile music_wav;
    private Integer genre;
    private String beat;

}