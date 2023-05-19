package com.a504.chatcomposer.user.dto.request;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;

import java.util.List;

@Schema(description = "회원정보 수정 요청 정보 DTO")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class UserUpdateReq {

    @Schema(description = "닉네임")
    private String nickname;

    @Schema(description = "선호 장르 목록")
    private List<Integer> favoriteGenres;
}
