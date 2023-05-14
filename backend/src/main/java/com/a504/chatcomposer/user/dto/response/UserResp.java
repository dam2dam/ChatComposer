package com.a504.chatcomposer.user.dto.response;

import com.a504.chatcomposer.music.dto.Member;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.music.entity.Music;
import com.a504.chatcomposer.user.entity.FavoriteMusic;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.querydsl.core.annotations.QueryProjection;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.*;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.util.CollectionUtils;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

import static com.a504.chatcomposer.global.util.Utils.*;

@Schema(description = "회원 정보 DTO")
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class UserResp {

	@Schema(description = "사용자 pk")
	private String userId;

	@Schema(description = "사용자 번호")
	private Long userSeq;

	@Schema(description = "닉네임")
	private String nickname;

	@Schema(description = "이메일")
	private String email;

	@Schema(description = "장르 목록")
	private List<Genre> favoriteGenres;

	@QueryProjection
	public UserResp(Long loginUserId, Music music, Long memberId, String nickname) {

	}
}
