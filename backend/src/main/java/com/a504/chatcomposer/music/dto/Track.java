package com.a504.chatcomposer.music.dto;

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
public class Track {

	@JsonProperty(value = "track_id")
	private Long trackId;

	@JsonProperty(value = "midi_description")
	private String midiDescription;

	@JsonProperty(value = "musical_instrument")
	private String musicalInstrument;
}
