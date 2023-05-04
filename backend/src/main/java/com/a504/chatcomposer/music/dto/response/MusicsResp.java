package com.a504.chatcomposer.music.dto.response;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

import org.springframework.util.CollectionUtils;

import com.a504.chatcomposer.member.entity.FavoriteMusic;
import com.a504.chatcomposer.music.dto.Member;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.a504.chatcomposer.music.entity.Music;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.querydsl.core.annotations.QueryProjection;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;
import lombok.extern.slf4j.Slf4j;

@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
@Slf4j
public class MusicsResp {

	public static final String NO = "n";
	public static final String YES = "y";
	@JsonProperty(value = "music_id")
	private long musicId;

	@JsonProperty(value = "member")
	private Member member;

	@JsonProperty(value = "title")
	private String title;

	@JsonProperty(value = "genre")
	private Genre genre;

	@JsonProperty(value = "tags")
	private List<String> tags;

	@JsonProperty(value = "favorite_count")
	private int favoriteCount;

	@JsonProperty(value = "created_at")
	private LocalDateTime createdAt;

	@JsonProperty(value = "cover_source")
	private String coverSource;

	@JsonProperty(value = "is_my_favorite")
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
		if (!CollectionUtils.isEmpty(favoriteMusics)) {
			for (int i = 0; i < favoriteMusics.size(); i++) {
				if (favoriteMusics.get(i).getMember().getId().equals(loginUserId)) {
					isMyFavorite = YES;
					break;
				}
			}
		}
		this.isMyFavorite = isMyFavorite;
	}
}
