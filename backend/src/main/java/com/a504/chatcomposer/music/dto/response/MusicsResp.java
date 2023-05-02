package com.a504.chatcomposer.music.dto.response;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

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

@Getter
@Setter
@Builder
@ToString
@NoArgsConstructor
@AllArgsConstructor
public class MusicsResp {

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

	@QueryProjection
	public MusicsResp(Music music, Long memberId, String nickname) {
		this.musicId = music.getId();
		this.title = music.getTitle();
		this.genre = music.getGenre();
		this.favoriteCount = music.getFavoriteCount();
		this.createdAt = music.getCreatedAt();
		this.coverSource = music.getCoverSource();
		
		this.member = Member.builder()
			.memberId(memberId)
			.nickname(nickname)
			.build();
		this.tags = music.getTags().stream()
			.map(m -> m.getTagName()).collect(Collectors.toList());
	}
}
