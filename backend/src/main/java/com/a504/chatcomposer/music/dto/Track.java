package com.a504.chatcomposer.music.dto;

import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Schema(description = "트랙 정보 DTO")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonNaming(value = PropertyNamingStrategies.SnakeCaseStrategy.class)
public class Track {

	@Schema(description = "트랙 pk")
	private Long trackId;

	@Schema(description = "음악을 재생하기 위한 midi 악보")
	private String midiDescription;

	@Schema(description = "악기명")
	private String musicalInstrument;
}
