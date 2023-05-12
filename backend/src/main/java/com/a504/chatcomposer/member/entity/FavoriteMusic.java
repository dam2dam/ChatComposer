package com.a504.chatcomposer.member.entity;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.Table;
import javax.validation.constraints.NotNull;

import com.a504.chatcomposer.music.entity.Music;

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
@Table(name = "favorite_music")
public class FavoriteMusic {

	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "favorite_music_id", nullable = false)
	private Long id;

	@NotNull
	@ManyToOne
	@JoinColumn(name = "user_seq")
	private User user;

	@NotNull
	@ManyToOne
	@JoinColumn(name = "music_id")
	private Music music;

}
