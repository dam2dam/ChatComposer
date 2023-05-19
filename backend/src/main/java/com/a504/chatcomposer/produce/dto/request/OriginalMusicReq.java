package com.a504.chatcomposer.produce.dto.request;

import lombok.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;

/**
 * 원본 음악을 저장하기 위한 API ([POST] /produce/cover) 요청에 필요한 리퀘스트 바디 정의.
 */
@Data
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
public class OriginalMusicReq {
    private String musicSource;
    private Integer genre;
    private ArrayList<String> moods;
    private ArrayList<String> instruements;
    private String musicPrompt;
    private String riffusionPrompt;

}