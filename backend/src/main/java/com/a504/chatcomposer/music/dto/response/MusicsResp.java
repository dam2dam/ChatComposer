package com.a504.chatcomposer.music.dto.response;

import java.time.LocalDateTime;
import java.util.List;

import com.a504.chatcomposer.music.dto.Member;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.fasterxml.jackson.annotation.JsonProperty;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@Builder
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
}
