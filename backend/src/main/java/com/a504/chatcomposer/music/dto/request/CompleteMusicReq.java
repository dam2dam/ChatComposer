package com.a504.chatcomposer.music.dto.request;

import java.util.List;

import org.springframework.web.multipart.MultipartFile;

import com.a504.chatcomposer.music.dto.Prompt;
import com.a504.chatcomposer.music.dto.Track;
import com.a504.chatcomposer.music.dto.enums.Beat;
import com.a504.chatcomposer.music.dto.enums.Genre;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

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
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class CompleteMusicReq {

	private MultipartFile title;

	private List<Integer> tags;

	private String description;

	private Genre genre;

	private Beat beat;

	private List<Track> tracks;

	private List<Prompt> prompts;

	private String musicSource;

	private String mixedMusicRequest;

	private String mixedMusic;

	private String coverRequest;

	private String coverSource;

	private int favoriteCount;
}
