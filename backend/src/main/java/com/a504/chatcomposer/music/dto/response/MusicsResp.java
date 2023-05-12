package com.a504.chatcomposer.music.dto.response;

import static com.a504.chatcomposer.global.util.Utils.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.util.CollectionUtils;

import com.a504.chatcomposer.user.entity.FavoriteMusic;
import com.a504.chatcomposer.music.dto.Member;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.music.entity.Music;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.querydsl.core.annotations.QueryProjection;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Schema(description = "음악 리스트를 위한 정보 DTO")
@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class MusicsResp {

	@Schema(description = "음악 pk")
	private long musicId;

	@Schema(description = "작성자 정보")
	private Member member;

	@Schema(description = "음악 제목")
	private String title;

	@Schema(description = "장르")
	private Genre genre;

	@Schema(description = "태그 목록")
	private List<String> tags;

	@Schema(description = "좋아요 수", defaultValue = "0")
	private int favoriteCount;

	@DateTimeFormat(pattern = "")
	@Schema(description = "음악 생성일")
	private LocalDateTime createdAt;

	@Schema(description = "앨범 커버 이미지의 S3 URL")
	private String coverSource;

	@Schema(description = "좋아요 여부", defaultValue = "n", allowableValues = {"y", "n"})
	private String isMyFavorite;

	@QueryProjection
	public MusicsResp(Long loginUserId, Music music, Long memberId, String nickname) {

		this.musicId = music.getId();
		this.title = music.getTitle();
		this.genre = music.getGenre();
		this.favoriteCount = music.getFavoriteCount();
		this.createdAt = music.getCreatedAt();
		this.coverSource = music.getCoverSource();

		// 작성자 정보
		this.member = Member.builder()
			.memberId(memberId)
			.nickname(nickname)
			.build();

		// music tags -> tag name 리스트로 만들기
		this.tags = music.getMusicTags().stream()
			.map(musicTag -> musicTag.getTag().getTagName()).collect(Collectors.toList());

		// 내가 좋아요 한 음악인지 여부
		String isMyFavorite = NO;
		List<FavoriteMusic> favoriteMusics = music.getFavoriteMusics();
		// 누군가 좋아요 한 음악이고 로그인 된 요청이라면 음악 좋아요 여부 판단
		if (!CollectionUtils.isEmpty(favoriteMusics) && loginUserId != NOT_LOGIN) {
			for (int i = 0; i < favoriteMusics.size(); i++) {
				if (favoriteMusics.get(i).getUser().getUserSeq().equals(loginUserId)) {
					isMyFavorite = YES;
					break;
				}
			}
		}
		// 아무도 좋아요를 누르지 않은 음악이거나 로그인 하지 않은 상태의 요청은 무조건 NO
		this.isMyFavorite = isMyFavorite;
	}
}
