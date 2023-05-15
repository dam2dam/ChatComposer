package com.a504.chatcomposer.music.entity;

import javax.persistence.CascadeType;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.OneToOne;
import javax.persistence.Table;
import javax.validation.constraints.NotNull;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Builder
@AllArgsConstructor
@NoArgsConstructor
@Getter
@Setter
@Entity
@Table(name = "track")
public class Track {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "track_id", nullable = false)
	private Long id;

	@NotNull
	@ManyToOne
	@JoinColumn(name = "music_id")
	private Music music;

	@Column(name = "midi_description", length = 500)
	private String midiDescription;

	@Column(name = "musical_instrument", length = 20)
	private String musicalInstrument;

	@OneToOne(mappedBy = "track", cascade = CascadeType.PERSIST)
	private Prompt prompt;

	/* 연관관계 메서드 설정 */
	public void setMusic(Music music) {
		this.music = music;
		music.getTracks().add(this);
	}
}
