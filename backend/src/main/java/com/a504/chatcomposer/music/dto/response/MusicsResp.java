package com.a504.chatcomposer.music.dto.response;

import static com.a504.chatcomposer.global.util.Utils.*;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.util.CollectionUtils;

import com.a504.chatcomposer.member.entity.FavoriteMusic;
import com.a504.chatcomposer.music.dto.Member;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.music.entity.Music;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import com.querydsl.core.annotations.QueryProjection;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class MusicsResp {

	private long musicId;

	private Member member;

	private String title;

	private Genre genre;

	private List<String> tags;

	private int favoriteCount;

	private LocalDateTime createdAt;

	private String coverSource;

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
			.map(m -> m.getTag().getTagName()).collect(Collectors.toList());

		// 내가 좋아요 한 음악인지 여부
		String isMyFavorite = NO;
		List<FavoriteMusic> favoriteMusics = music.getFavoriteMusics();
		// 누군가 좋아요 한 음악이고 로그인 된 요청이라면 음악 좋아요 여부 판단
		if (!CollectionUtils.isEmpty(favoriteMusics) && loginUserId != NOT_LOGIN) {
			for (int i = 0; i < favoriteMusics.size(); i++) {
				if (favoriteMusics.get(i).getMember().getId().equals(loginUserId)) {
					isMyFavorite = YES;
					break;
				}
			}
		}
		// 아무도 좋아요를 누르지 않은 음악이거나 로그인 하지 않은 상태의 요청은 무조건 NO
		this.isMyFavorite = isMyFavorite;
	}
}
