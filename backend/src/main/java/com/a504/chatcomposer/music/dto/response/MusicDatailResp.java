package com.a504.chatcomposer.music.dto.response;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.web.multipart.MultipartFile;

import com.a504.chatcomposer.music.dto.Member;
import com.a504.chatcomposer.music.dto.Prompt;
import com.a504.chatcomposer.music.dto.Track;
import com.a504.chatcomposer.music.dto.enums.Beat;
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
public class MusicDatailResp {

	@JsonProperty(value = "music_id")
	private long musicId;

	@JsonProperty(value = "member")
	private Member member;

	@JsonProperty(value = "title")
	private MultipartFile title;

	@JsonProperty(value = "tags")
	private List<Integer> tags;

	@JsonProperty(value = "description")
	private String description;

	@JsonProperty(value = "genre")
	private Genre genre;

	@JsonProperty(value = "beat")
	private Beat beat;

	@JsonProperty(value = "created_at")
	private LocalDateTime createdAt;

	@JsonProperty(value = "tracks")
	private List<Track> tracks;

	@JsonProperty(value = "prompts")
	private List<Prompt> prompts;

	@JsonProperty(value = "music_source")
	private String musicSource;

	@JsonProperty(value = "mixed_music_request")
	private String mixedMusicRequest;

	@JsonProperty(value = "mixed_music")
	private String mixedMusic;

	@JsonProperty(value = "cover_request")
	private String coverRequest;

	@JsonProperty(value = "cover_source")
	private String coverSource;

	@JsonProperty(value = "favorite_count")
	private int favoriteCount;
}
